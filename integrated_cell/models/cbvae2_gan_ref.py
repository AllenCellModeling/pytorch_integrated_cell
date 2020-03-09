import torch
import numpy as np
import pickle

from .. import utils
from . import cbvae2_gan

from integrated_cell.model_utils import tensor2img
from integrated_cell.utils import plots as plots

import scipy


class Model(cbvae2_gan.Model):
    def __init__(self, **kwargs):

        super(Model, self).__init__(**kwargs)

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
            x, classes, x_mesh = self.data_provider.get_sample()
            x = x.cuda(gpu_id)
            x_mesh = x_mesh.cuda(gpu_id)

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
                zAll, z_mesh = enc(x, classes_onehot, x_mesh)

            for i in range(len(zAll)):
                zAll[i] = self.reparameterize(zAll[i][0], zAll[i][1])
                zAll[i].detach_()

            with torch.no_grad():
                xHat = dec([classes_onehot] + zAll, z_mesh)

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
                xHat = dec([classes_onehot] + zAll, z_mesh)

            yHat_xFake2 = decD(xHat)
            errDecD_fake2 = crit_decD(yHat_xFake2, y_xFake)

            decDLoss_tmp = (errDecD_real + (errDecD_fake + errDecD_fake2) / 2) / 2
            decDLoss_tmp.backward()

            opt_decD.step()

            decDLoss += decDLoss_tmp.item()

        errDecD_real = None
        errDecD_fake = None
        errDecD_fake2 = None

        for p in enc.parameters():
            p.requires_grad = True

        for p in dec.parameters():
            p.requires_grad = True

        for p in decD.parameters():
            p.requires_grad = False

        opt_enc.zero_grad()
        opt_dec.zero_grad()
        opt_decD.zero_grad()

        #####################
        # train autoencoder
        #####################
        torch.cuda.empty_cache()

        # Forward passes
        [z_ref, z_struct], z_mesh = enc(x, classes_onehot, x_mesh)

        kld_ref = self.kld_loss(z_ref[0], z_ref[1])
        kld_struct = self.kld_loss(z_struct[0], z_struct[1])

        kld_loss = kld_ref + kld_struct

        kld_loss_ref = kld_ref.item()
        kld_loss_struct = kld_struct.item()

        zLatent = z_struct[0].data.cpu()

        zAll = [z_ref, z_struct]
        for i in range(len(zAll)):
            zAll[i] = self.reparameterize(zAll[i][0], zAll[i][1])

        xHat = dec([classes_onehot] + zAll, z_mesh)

        # resample from the structure space and make sure that the reference channel stays the same
        # shuffle_inds = torch.randperm(x.shape[0])
        # zAll[-1].normal_()
        # xHat2 = dec([classes_onehot[shuffle_inds]] + zAll)

        # Update the image reconstruction
        recon_loss = crit_recon(xHat, x)

        beta_vae_loss = self.vae_loss(recon_loss, kld_loss)

        beta_vae_loss.backward(retain_graph=True)

        recon_loss = recon_loss.item()

        opt_enc.step()

        if self.lambda_decD_loss > 0:
            for p in enc.parameters():
                p.requires_grad = False

            # update wrt decD(dec(enc(X)))
            yHat_xFake = decD(xHat)
            minimaxDecDLoss = crit_decD(yHat_xFake, y_xReal)

            shuffle_inds = torch.randperm(x.shape[0])
            xHat = dec([classes_onehot[shuffle_inds]] + zAll, z_mesh)

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

    def save_progress(self):
        gpu_id = self.gpu_ids[0]
        epoch = self.get_current_epoch()

        data_provider = self.data_provider
        enc = self.enc
        dec = self.dec

        enc.train(False)
        dec.train(False)

        ###############
        # TRAINING DATA
        ###############
        train_classes = data_provider.get_classes(
            np.arange(0, data_provider.get_n_dat("train", override=True)), "train"
        )
        _, train_inds = np.unique(train_classes.numpy(), return_index=True)

        x, classes, x_mesh = data_provider.get_sample(
            "train", train_inds, patched=False
        )
        x = x.cuda(gpu_id)

        classes = classes.type_as(x).long()
        classes_onehot = utils.index_to_onehot(
            classes, self.data_provider.get_n_classes()
        )

        x_mesh = x_mesh.type_as(x)

        def xHat2sample(xHat, x):
            if xHat.shape[1] == x.shape[1]:
                pass
            else:
                mu = xHat[:, 0::2, :, :]
                log_var = torch.log(xHat[:, 1::2, :, :])

                xHat = self.reparameterize(mu, log_var, add_noise=True)

            return xHat

        with torch.no_grad():
            z, z_mesh = enc(x, classes_onehot, x_mesh)
            for i in range(len(z)):
                z[i] = z[i][0]
            xHat = dec([classes_onehot] + z, z_mesh)
            xHat = xHat2sample(xHat, x)

        imgX = tensor2img(x.data.cpu())
        imgXHat = tensor2img(xHat.data.cpu())
        imgTrainOut = np.concatenate((imgX, imgXHat), 0)

        ###############
        # TESTING DATA
        ###############
        test_classes = data_provider.get_classes(
            np.arange(0, data_provider.get_n_dat("test")), "test"
        )
        _, test_inds = np.unique(test_classes.numpy(), return_index=True)

        x, classes, ref = data_provider.get_sample("test", test_inds, patched=False)
        x = x.cuda(gpu_id)
        classes = classes.type_as(x).long()
        ref = ref.type_as(x)

        with torch.no_grad():
            z, z_mesh = enc(x, classes_onehot, x_mesh)
            for i in range(len(z)):
                z[i] = z[i][0]

            xHat = dec([classes_onehot] + z, z_mesh)
            xHat = xHat2sample(xHat, x)

        for z_sub in z:
            z_sub.normal_()

        with torch.no_grad():
            xHat_z = dec([classes_onehot] + z, z_mesh)
            xHat_z = xHat2sample(xHat_z, x)

        imgX = tensor2img(x.data.cpu())
        imgXHat = tensor2img(xHat.data.cpu())
        imgXHat_z = tensor2img(xHat_z.data.cpu())
        imgTestOut = np.concatenate((imgX, imgXHat, imgXHat_z), 0)

        imgOut = np.concatenate((imgTrainOut, imgTestOut))

        scipy.misc.imsave(
            "{0}/progress_{1}.png".format(self.save_dir, int(epoch - 1)), imgOut
        )

        embeddings_train = np.concatenate(self.zAll, 0)

        pickle.dump(
            embeddings_train, open("{0}/embedding.pth".format(self.save_dir), "wb")
        )
        pickle.dump(
            embeddings_train,
            open(
                "{0}/embedding_{1}.pth".format(self.save_dir, self.get_current_iter()),
                "wb",
            ),
        )

        pickle.dump(self.logger, open("{0}/logger_tmp.pkl".format(self.save_dir), "wb"))

        # History
        plots.history(self.logger, "{0}/history.png".format(self.save_dir))

        # Short History
        plots.short_history(self.logger, "{0}/history_short.png".format(self.save_dir))

        # Embedding figure
        plots.embeddings(embeddings_train, "{0}/embedding.png".format(self.save_dir))

        # embeddings_validate = embeddings.get_latent_embeddings(
        #     enc,
        #     dec,
        #     dp=self.data_provider,
        #     recon_loss=self.crit_recon,
        #     modes=["validate"],
        #     batch_size=self.data_provider.batch_size,
        # )
        # embeddings_validate["iteration"] = self.get_current_iter()
        # embeddings_validate["epoch"] = self.get_current_epoch()

        # torch.save(
        #     embeddings_validate,
        #     "{}/embeddings_validate_{}.pth".format(
        #         self.save_dir, self.get_current_iter()
        #     ),
        # )

        xHat = None
        x = None

        enc.train(True)
        dec.train(True)
