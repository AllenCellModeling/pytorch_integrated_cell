import torch
import numpy as np
import os
import pickle
import shutil

from .. import utils
from . import cbvae2
from .. import SimpleLogger
from .. import model_utils


# Like cbvae2_gan but we update in a single pass


class Model(cbvae2.Model):
    def __init__(
        self, decD, opt_decD, crit_decD, lambda_decD_loss=1, n_decD_steps=1, **kwargs
    ):

        super(Model, self).__init__(**kwargs)

        self.decD = decD
        self.opt_decD = opt_decD

        self.crit_decD = crit_decD

        self.lambda_decD_loss = lambda_decD_loss
        self.n_decD_steps = n_decD_steps

        logger_path = "{}/logger.pkl".format(self.save_dir)
        if os.path.exists(logger_path):
            self.logger = pickle.load(open(logger_path, "rb"))
        else:
            columns = ("epoch", "iter", "reconLoss")
            print_str = "[%d][%d] reconLoss: %.6f"

            columns += (
                "kldLossRef",
                "kldLossStruct",
                "minimaxDecDLoss",
                "decDLoss",
                "time",
            )
            print_str += (
                " kld ref: %.6f kld struct: %.6f mmDecD: %.6f decD: %.6f time: %.2f"
            )
            self.logger = SimpleLogger(columns, print_str)

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
            p.requires_grad = True

        for p in dec.parameters():
            p.requires_grad = True

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

            zAll = enc(x, classes_onehot)

            for i in range(len(zAll)):
                zAll[i] = self.reparameterize(zAll[i][0], zAll[i][1])
                # zAll[i].detach_()

            # with torch.no_grad():
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

            # with torch.no_grad():
            xHat = dec([classes_onehot] + zAll)

            yHat_xFake2 = decD(xHat)
            errDecD_fake2 = crit_decD(yHat_xFake2, y_xFake)

            decDLoss_tmp = (errDecD_real + (errDecD_fake + errDecD_fake2) / 2) / 2
            decDLoss_tmp.backward()

            opt_decD.step()

            decDLoss += decDLoss_tmp.item()

            #####################
            # train autoencoder
            #####################
            # Forward passes
            z_ref, z_struct = zAll

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

            # resample from the structure space and make sure that the reference channel stays the same
            # shuffle_inds = torch.randperm(x.shape[0])
            # zAll[-1].normal_()
            # xHat2 = dec([classes_onehot[shuffle_inds]] + zAll)

            # Update the image reconstruction
            recon_loss = crit_recon(xHat, x)

            beta_vae_loss = self.vae_loss(recon_loss, kld_loss)

            beta_vae_loss.backward()

            recon_loss = recon_loss.item()

            opt_dec.step()
            opt_enc.step()

        if self.lambda_decD_loss > 0:
            opt_dec.zero_grad()
            opt_enc.zero_grad()
            opt_decD.zero_grad()

            # update wrt decD(dec(enc(X)))
            yHat_xFake = decD(xHat)
            minimaxDecDLoss = crit_decD(yHat_xFake, y_xReal)

            shuffle_inds = torch.randperm(x.shape[0])
            xHat = dec([classes_onehot[shuffle_inds]] + zAll)

            yHat_xFake2 = decD(xHat)
            minimaxDecDLoss2 = crit_decD(yHat_xFake2, y_xReal[shuffle_inds])

            minimaxDecLoss = (minimaxDecDLoss + minimaxDecDLoss2) / 2
            minimaxDecLoss.mul(self.lambda_decD_loss).backward()
            minimaxDecLoss = minimaxDecLoss.item()
        else:
            minimaxDecLoss = 0

        opt_dec.step()

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
