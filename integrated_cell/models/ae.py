import torch
import numpy as np
from . import base_model
from .. import SimpleLogger

import scipy

from ..utils.plots import tensor2im
from integrated_cell.utils import plots as plots
from .. import model_utils

import os
import pickle

import shutil


class Model(base_model.Model):
    def __init__(
        self, enc, dec, opt_enc, opt_dec, crit_recon, n_display_imgs=10, **kwargs
    ):
        # Train an autoencoder

        super(Model, self).__init__(**kwargs)

        self.enc = enc
        self.dec = dec
        self.opt_enc = opt_enc
        self.opt_dec = opt_dec

        self.crit_recon = crit_recon

        self.n_display_imgs = n_display_imgs

        logger_path = "{}/logger.pkl".format(self.save_dir)

        if os.path.exists(logger_path):
            self.logger = pickle.load(open(logger_path, "rb"))
        else:
            columns = ("epoch", "iter", "reconLoss", "time")
            print_str = "[%d][%d] reconLoss: %.6f time: %.2f"
            self.logger = SimpleLogger(columns, print_str)

    def iteration(self):

        torch.cuda.empty_cache()

        enc, dec = self.enc, self.dec
        opt_enc, opt_dec = self.opt_enc, self.opt_dec
        crit_recon = self.crit_recon

        enc.train(True)
        dec.train(True)

        for p in enc.parameters():
            p.requires_grad = True

        for p in dec.parameters():
            p.requires_grad = True

        # Ignore class labels and reference information
        x, _, _ = self.data_provider.get_sample()
        x = x.cuda()

        opt_enc.zero_grad()
        opt_dec.zero_grad()

        #####################
        # train autoencoder
        #####################

        # Forward passes
        z = enc(x)

        xHat = dec(z)

        # Update the image reconstruction
        recon_loss = crit_recon(xHat, x)

        recon_loss.backward()

        opt_enc.step()
        opt_dec.step()

        zLatent = z.data.cpu()
        recon_loss = recon_loss.item()

        errors = [recon_loss]

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

        x, _, _ = data_provider.get_sample("train", img_inds)
        x = x.cuda(gpu_id)

        with torch.no_grad():
            z = enc(x)
            xHat = dec(z)

        imgX = tensor2im(x.data.cpu())
        imgXHat = tensor2im(xHat.data.cpu())
        imgTrainOut = np.concatenate((imgX, imgXHat), 0)

        ###############
        # TESTING DATA
        ###############
        x, _, _ = data_provider.get_sample("validate", img_inds)
        x = x.cuda(gpu_id)

        with torch.no_grad():
            z = enc(x)
            xHat = dec(z)

        imgX = tensor2im(x.data.cpu())
        imgXHat = tensor2im(xHat.data.cpu())

        imgTestOut = np.concatenate((imgX, imgXHat), 0)

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
