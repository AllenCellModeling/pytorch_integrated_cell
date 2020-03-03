import torch
import numpy as np
from .. import utils
from . import cbvae2

import scipy

from integrated_cell.model_utils import tensor2img
from integrated_cell.utils import plots as plots
from .. import model_utils
from ..metrics import embeddings

import pickle

import shutil

# conditional beta variational autencoder
def reparameterize(mu, log_var, add_noise=True):
    if add_noise:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        out = mu + eps * std
    else:
        out = mu

    return out


class Model(cbvae2.Model):
    def __init__(self, prog_model_dir, prog_parent_dir, prog_suffix, **kwargs):

        super(Model, self).__init__(**kwargs)

        networks, _, _ = utils.load_network_from_dir(
            prog_model_dir, parent_dir=prog_parent_dir, suffix=prog_suffix
        )

        self.prog_enc = networks["enc"]
        self.prog_enc.train(False)

        for p in self.prog_enc.parameters():
            p.requires_grad = False

        self.prog_dec = networks["dec"]
        self.prog_dec.train(False)

        for p in self.prog_dec.parameters():
            p.requires_grad = False

    def iteration(self):

        torch.cuda.empty_cache()

        enc, dec = self.enc, self.dec
        opt_enc, opt_dec = self.opt_enc, self.opt_dec
        crit_recon = self.crit_recon

        # do this just incase anything upstream changes these values
        enc.train(True)
        dec.train(True)

        x, classes, ref = self.data_provider.get_sample()
        x = x.cuda()

        classes = classes.type_as(x).long()
        classes_onehot = utils.index_to_onehot(
            classes, self.data_provider.get_n_classes()
        )

        for p in enc.parameters():
            p.requires_grad = True

        for p in dec.parameters():
            p.requires_grad = True

        opt_enc.zero_grad()
        opt_dec.zero_grad()

        #####################
        # train autoencoder
        #####################

        # Forward passes
        with torch.no_grad():
            x_ref, x_struct = self.prog_enc(x, classes_onehot)

        # take only the mus
        x_ref, x_struct = x_ref[0].detach(), x_struct[0].detach()

        z_ref, z_struct = enc(x_ref, x_struct, classes_onehot)

        kld_ref = self.kld_loss(z_ref[0], z_ref[1])
        kld_struct = self.kld_loss(z_struct[0], z_struct[1])

        kld_loss = kld_ref + kld_struct

        kld_loss_ref = kld_ref.item()
        kld_loss_struct = kld_struct.item()

        zLatent = z_struct[0].data.cpu()

        zAll = [z_ref, z_struct]
        for i in range(len(zAll)):
            zAll[i] = self.reparameterize(zAll[i][0], zAll[i][1])

        x_hat_ref, x_hat_struct = dec([classes_onehot] + zAll)

        # Update the image reconstruction
        recon_loss = crit_recon(x_hat_ref, x_ref) + crit_recon(x_hat_struct, x_struct)

        beta_vae_loss = self.vae_loss(recon_loss, kld_loss)

        beta_vae_loss.backward()

        recon_loss = recon_loss.item()

        opt_enc.step()
        opt_dec.step()

        errors = [recon_loss, kld_loss_ref, kld_loss_struct]

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

        x, classes, ref = data_provider.get_sample("train", train_inds)
        x = x.cuda(gpu_id)

        classes = classes.type_as(x).long()
        classes_onehot = utils.index_to_onehot(
            classes, self.data_provider.get_n_classes()
        )

        ref = ref.type_as(x)

        def xHat2sample(xHat, x):
            if xHat.shape[1] == x.shape[1]:
                pass
            else:
                mu = xHat[:, 0::2, :, :]
                log_var = torch.log(xHat[:, 1::2, :, :])

                xHat = reparameterize(mu, log_var, add_noise=True)

            return xHat

        with torch.no_grad():
            z = enc(x, classes_onehot)
            for i in range(len(z)):
                z[i] = z[i][0]
            xHat = dec([classes_onehot] + z)
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

        x, classes, ref = data_provider.get_sample("test", test_inds)
        x = x.cuda(gpu_id)
        classes = classes.type_as(x).long()
        ref = ref.type_as(x)

        with torch.no_grad():
            z = enc(x, classes_onehot)
            for i in range(len(z)):
                z[i] = z[i][0]

            xHat = dec([classes_onehot] + z)
            xHat = xHat2sample(xHat, x)

        for z_sub in z:
            z_sub.normal_()

        with torch.no_grad():
            xHat_z = dec([classes_onehot] + z)
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

        embeddings_validate = embeddings.get_latent_embeddings(
            enc,
            dec,
            dp=self.data_provider,
            recon_loss=self.crit_recon,
            modes=["validate"],
            batch_size=self.data_provider.batch_size,
        )
        embeddings_validate["iteration"] = self.get_current_iter()
        embeddings_validate["epoch"] = self.get_current_epoch()

        torch.save(
            embeddings_validate,
            "{}/embeddings_validate_{}.pth".format(
                self.save_dir, self.get_current_iter()
            ),
        )

        xHat = None
        x = None

        enc.train(True)
        dec.train(True)

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

        model_utils.save_state(self.enc, self.opt_enc, enc_save_path_tmp, gpu_id)
        shutil.copyfile(enc_save_path_tmp, enc_save_path_final)

        model_utils.save_state(self.dec, self.opt_dec, dec_save_path_tmp, gpu_id)
        shutil.copyfile(dec_save_path_tmp, dec_save_path_final)

        pickle.dump(self.logger, open("{0}/logger.pkl".format(save_dir), "wb"))
        pickle.dump(
            self.logger, open("{0}/logger_{1}.pkl".format(save_dir, n_iters), "wb")
        )
