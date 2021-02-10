import torch
import numpy as np
from . import cbvae2
from .. import SimpleLogger

import scipy

from ..utils.plots import tensor2im
from integrated_cell.utils import plots as plots
from ..utils import reparameterize
from .. import model_utils

from ..metrics import embeddings_reference as embeddings

import os
import pickle

import shutil


def kl_divergence(mu, logvar):

    batch_size = mu.size(0)
    assert batch_size != 0

    if mu.data.ndimension() >= 4:
        mu = mu.view(mu.size(0), -1)

    if logvar.data.ndimension() >= 4:
        logvar = logvar.view(logvar.size(0), -1)

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class Model(cbvae2.Model):
    def __init__(self, n_display_imgs=10, **kwargs):

        super(Model, self).__init__(**kwargs)

        self.n_display_imgs = n_display_imgs

        logger_path = "{}/logger.pkl".format(self.save_dir)
        if os.path.exists(logger_path):
            self.logger = pickle.load(open(logger_path, "rb"))
        else:
            columns = ("epoch", "iter", "reconLoss")
            print_str = "[%d][%d] reconLoss: %.6f"

            columns += ("kldLoss", "time")
            print_str += " kld: %.6f time: %.2f"
            self.logger = SimpleLogger(columns, print_str)

    def reparameterize(self, mu, log_var, *args):
        return reparameterize(mu, log_var, *args)

    def kld_loss(self, mu, log_var):
        return kl_divergence(mu, log_var)

    def iteration(self):

        torch.cuda.empty_cache()

        enc, dec = self.enc, self.dec
        opt_enc, opt_dec = self.opt_enc, self.opt_dec
        crit_recon = self.crit_recon

        # do this just incase anything upstream changes these values
        enc.train(True)
        dec.train(True)

        x = self.data_provider.get_sample()
        x = x.cuda()

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
        mu, logsigma = enc(x)

        kld_loss = self.kld_loss(mu, logsigma)

        zLatent = mu.data.cpu()

        z = self.reparameterize(mu, logsigma)

        xHat = dec(z)

        # Update the image reconstruction
        recon_loss = crit_recon(xHat, x)

        beta_vae_loss = self.vae_loss(recon_loss, kld_loss)

        beta_vae_loss.backward()

        recon_loss = recon_loss.item()

        opt_enc.step()
        opt_dec.step()

        errors = [recon_loss, kld_loss.item()]

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
        img_inds = np.arange(self.n_display_imgs)

        x = data_provider.get_sample("train", img_inds)
        x = x.cuda(gpu_id)

        def xHat2sample(xHat, x):
            if xHat.shape[1] == x.shape[1]:
                pass
            else:
                mu = xHat[:, 0::2, :, :]
                log_var = torch.log(xHat[:, 1::2, :, :])

                xHat = self.reparameterize(mu, log_var, add_noise=True)

            return xHat

        with torch.no_grad():
            z_mu, _ = enc(x)
            xHat = dec(z_mu)
            xHat = xHat2sample(xHat, x)

        imgX = tensor2im(x.data.cpu())
        imgXHat = tensor2im(xHat.data.cpu())
        imgTrainOut = np.concatenate((imgX, imgXHat), 0)

        ###############
        # TESTING DATA
        ###############
        x = data_provider.get_sample("validate", img_inds)
        x = x.cuda(gpu_id)

        with torch.no_grad():
            z_mu, _ = enc(x)
            xHat = dec(z_mu)
            xHat = xHat2sample(xHat, x)

        z_mu.normal_()

        with torch.no_grad():
            xHat_z = dec(z_mu)
            xHat_z = xHat2sample(xHat_z, x)

        imgX = tensor2im(x.data.cpu())
        imgXHat = tensor2im(xHat.data.cpu())
        imgXHat_z = tensor2im(xHat_z.data.cpu())
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

        def sampler(mode, inds):
            return data_provider.get_sample(mode, inds)

        embeddings_validate = embeddings.get_latent_embeddings(
            enc,
            dec,
            dp=self.data_provider,
            recon_loss=self.crit_recon,
            modes=["validate"],
            batch_size=self.data_provider.batch_size,
            sampler=sampler,
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

        img_embeddings = np.concatenate(self.zAll, 0)
        pickle.dump(img_embeddings, open("{0}/embedding.pth".format(save_dir), "wb"))
        pickle.dump(
            img_embeddings,
            open("{0}/embedding_{1}.pth".format(save_dir, n_iters), "wb"),
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