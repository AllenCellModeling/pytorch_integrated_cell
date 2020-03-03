import torch
import numpy as np
import pickle
import shutil

from .. import utils
from . import cbvae2_gan
from .. import model_utils

from apex import amp


class Model(cbvae2_gan.Model):
    def __init__(self, opt_level="O1", **kwargs):

        super(Model, self).__init__(**kwargs)

        is_data_parallel = False
        # do some hack if set up for DataParallel
        if (
            str(self.enc.__class__)
            == "<class 'torch.nn.parallel.data_parallel.DataParallel'>"
        ):
            is_data_parallel = True
            device_ids_enc = self.enc.device_ids
            device_ids_dec = self.dec.device_ids
            device_ids_decD = self.decD.device_ids

            self.enc = self.enc.module
            self.dec = self.dec.module
            self.decD = self.decD.module

        [self.enc, self.dec, self.decD], [
            self.opt_enc,
            self.opt_dec,
            self.opt_decD,
        ] = amp.initialize(
            [self.enc, self.dec, self.decD],
            [self.opt_enc, self.opt_dec, self.opt_decD],
            opt_level=opt_level,
            num_losses=3,
        )

        if is_data_parallel:
            self.enc = torch.nn.DataParallel(self.enc, device_ids_enc)
            self.dec = torch.nn.DataParallel(self.dec, device_ids_dec)
            self.decD = torch.nn.DataParallel(self.dec, device_ids_decD)

    def iteration(self):

        torch.cuda.empty_cache()

        gpu_id = self.gpu_ids[0]

        enc, dec, decD = self.enc, self.dec, self.decD
        opt_enc, opt_dec, opt_decD = self.opt_enc, self.opt_dec, self.opt_decD
        crit_recon, crit_decD = (self.crit_recon, self.crit_decD)

        # do this just incase anything upstream changes these values
        enc.train(True)
        dec.train(True)
        decD.train(True)

        # update the discriminator
        # maximize log(AdvZ(z)) + log(1 - AdvZ(Enc(x)))

        for p in decD.parameters():
            p.requires_grad = True

        for p in enc.parameters():
            p.requires_grad = False

        for p in dec.parameters():
            p.requires_grad = False

        decDLoss = 0
        for step_num in range(self.n_decD_steps):
            x, classes, ref = self.data_provider.get_sample()
            x = x.cuda(gpu_id)

            classes = classes.type_as(x).long()
            classes_onehot = utils.index_to_onehot(
                classes, self.data_provider.get_n_classes()
            )

            y_xReal = classes
            y_xFake = (
                torch.zeros(classes.shape)
                .fill_(self.data_provider.get_n_classes())
                .type_as(x)
                .long()
            )
            with torch.no_grad():
                zAll = enc(x, classes_onehot)

            for i in range(len(zAll)):
                zAll[i] = self.reparameterize(zAll[i][0], zAll[i][1])
                zAll[i].detach_()

            with torch.no_grad():
                xHat = dec([classes_onehot] + zAll)

            opt_enc.zero_grad()
            opt_dec.zero_grad()
            opt_decD.zero_grad()

            ##############
            # Train decD
            ##############

            # train with real
            yHat_xReal = decD(x)
            errDecD_real = crit_decD(yHat_xReal, y_xReal)

            # train with fake, reconstructed
            yHat_xFake = decD(xHat)
            errDecD_fake = crit_decD(yHat_xFake, y_xFake)

            # train with fake, sampled and decoded
            for z in zAll:
                z.normal_()

            with torch.no_grad():
                xHat = dec([classes_onehot] + zAll)

            yHat_xFake2 = decD(xHat)
            errDecD_fake2 = crit_decD(yHat_xFake2, y_xFake)

            decDLoss_tmp = (errDecD_real + (errDecD_fake + errDecD_fake2) / 2) / 2

            with amp.scale_loss(decDLoss_tmp, [opt_decD], loss_id=0) as scaled_loss:
                scaled_loss.backward()

            opt_decD.step()

            decDLoss += decDLoss_tmp.item()

        opt_enc.zero_grad()
        opt_dec.zero_grad()
        opt_decD.zero_grad()

        for p in enc.parameters():
            p.requires_grad = True

        for p in dec.parameters():
            p.requires_grad = True

        for p in decD.parameters():
            p.requires_grad = False

        #####################
        # train autoencoder
        #####################
        torch.cuda.empty_cache()

        # Forward passes
        z_ref, z_struct = enc(x, classes_onehot)

        kld_ref = self.kld_loss(z_ref[0], z_ref[1])
        kld_struct = self.kld_loss(z_struct[0], z_struct[1])

        kld_loss = kld_ref + kld_struct

        kld_loss_ref = kld_ref.item()
        kld_loss_struct = kld_struct.item()

        zLatent = z_struct[0].data.cpu()

        zAll = [z_ref, z_struct]
        for i in range(len(zAll)):
            zAll[i] = self.reparameterize(zAll[i][0], zAll[i][1])

        xHat = dec([classes_onehot] + zAll)

        recon_loss = crit_recon(xHat, x)

        beta_vae_loss = self.vae_loss(recon_loss, kld_loss)

        if self.lambda_decD_loss > 0:
            for p in enc.parameters():
                p.requires_grad = False

            # update wrt decD(dec(enc(X)))
            yHat_xFake = decD(xHat)
            minimaxDecDLoss = crit_decD(yHat_xFake, y_xReal)

            shuffle_inds = torch.randperm(x.shape[0])
            xHat = dec([classes_onehot[shuffle_inds]] + zAll)

            yHat_xFake2 = decD(xHat)
            minimaxDecDLoss2 = crit_decD(yHat_xFake2, y_xReal[shuffle_inds])

            minimaxDecLoss = (minimaxDecDLoss + minimaxDecDLoss2) / 2
            minimaxDecLoss.mul(self.lambda_decD_loss).backward()

            with amp.scale_loss(
                minimaxDecLoss + beta_vae_loss, [opt_dec, opt_enc], loss_id=1
            ) as scaled_loss:
                scaled_loss.backward()

            minimaxDecLoss = minimaxDecLoss.item()
        else:
            minimaxDecLoss = 0

        opt_dec.step()
        opt_enc.step()

        recon_loss = recon_loss.item()

        errors = [recon_loss, kld_loss_ref, kld_loss_struct, minimaxDecLoss, decDLoss]

        return errors, zLatent

    def save(self, save_dir):
        #         for saving and loading see:
        #         https://discuss.pytorch.org/t/how-to-save-load-torch-models/718

        gpu_id = self.gpu_ids[0]

        n_iters = self.get_current_iter()

        embeddings = np.concatenate(self.zAll, 0)
        pickle.dump(embeddings, open("{0}/embedding.pth".format(save_dir), "wb"))
        pickle.dump(
            embeddings, open("{0}/embedding_{1}.pth".format(save_dir, n_iters), "wb")
        )

        enc_save_path_tmp = "{0}/enc.pth".format(save_dir)
        enc_save_path_final = "{0}/enc_{1}.pth".format(save_dir, n_iters)
        dec_save_path_tmp = "{0}/dec.pth".format(save_dir)
        dec_save_path_final = "{0}/dec_{1}.pth".format(save_dir, n_iters)

        decD_save_path_tmp = "{0}/decD.pth".format(save_dir)
        decD_save_path_final = "{0}/decD_{1}.pth".format(save_dir, n_iters)

        model_utils.save_state(self.enc, self.opt_enc, enc_save_path_tmp, gpu_id)
        shutil.copyfile(enc_save_path_tmp, enc_save_path_final)

        model_utils.save_state(self.dec, self.opt_dec, dec_save_path_tmp, gpu_id)
        shutil.copyfile(dec_save_path_tmp, dec_save_path_final)

        model_utils.save_state(self.decD, self.opt_decD, decD_save_path_tmp, gpu_id)
        shutil.copyfile(decD_save_path_tmp, decD_save_path_final)

        pickle.dump(self.logger, open("{0}/logger.pkl".format(save_dir), "wb"))
        pickle.dump(
            self.logger, open("{0}/logger_{1}.pkl".format(save_dir, n_iters), "wb")
        )
