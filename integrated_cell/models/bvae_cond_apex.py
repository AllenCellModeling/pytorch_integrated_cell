import torch
import pickle
import os

import numpy as np
import scipy

from apex import amp

from .. import model_utils
from .. import utils
from . import bvae
from .. import SimpleLogger
from . import base_model

from integrated_cell.model_utils import tensor2img
from integrated_cell.utils import plots as plots


# This is the trainer for the conditional Beta-VAE


def reparameterize(mu, log_var, add_noise=True):
    if add_noise:
        std = log_var.div(2).exp()
        eps = torch.randn_like(std)
        out = eps.mul(std).add_(mu)
    else:
        out = mu

    return out


class Model(base_model.Model):
    def __init__(
        self,
        enc,
        dec,
        opt_enc,
        opt_dec,
        n_epochs,
        gpu_ids,
        save_dir,
        data_provider,
        crit_recon,
        crit_z_class=None,
        crit_z_ref=None,
        save_progress_iter=1,
        save_state_iter=10,
        beta=1,
        c_max=25,
        c_iters_max=1.2e5,
        gamma=1000,
        objective="H",
        lambda_ref_loss=1,
        lambda_class_loss=1,
        provide_decoder_vars=False,
        kld_avg=False,
        opt_level="O1",
    ):

        super(Model, self).__init__(
            data_provider,
            n_epochs,
            gpu_ids,
            save_dir=save_dir,
            save_state_iter=save_state_iter,
            save_progress_iter=save_progress_iter,
            provide_decoder_vars=provide_decoder_vars,
        )

        [enc, dec], [opt_enc, opt_dec] = amp.initialize(
            [enc, dec], [opt_enc, opt_dec], opt_level=opt_level
        )

        enc = torch.nn.parallel.DataParallel(enc, gpu_ids)
        dec = torch.nn.parallel.DataParallel(dec, gpu_ids)

        self.enc = enc
        self.dec = dec

        self.opt_enc = opt_enc
        self.opt_dec = opt_dec

        self.crit_recon = crit_recon
        self.crit_z_class = crit_z_class
        self.crit_z_ref = crit_z_ref

        self.kld_avg = kld_avg

        if objective == "H":
            self.beta = beta
        elif objective == "B":
            self.c_max = c_max
            self.gamma = gamma
            self.c_iters_max = c_iters_max

        self.objective = objective

        self.lambda_ref_loss = lambda_ref_loss
        self.lambda_class_loss = lambda_class_loss

        logger_path = "{}/logger.pkl".format(save_dir)
        if os.path.exists(logger_path):
            self.logger = pickle.load(open(logger_path, "rb"))
        else:
            columns = ("epoch", "iter", "reconLoss")
            print_str = "[%d][%d] reconLoss: %.6f"

            columns += ("kldLossRef", "kldLossStruct", "time")
            print_str += " kld ref: %.6f kld struct: %.6f time: %.2f"
            self.logger = SimpleLogger(columns, print_str)

    def iteration(self):

        gpu_id = self.gpu_ids[0]

        enc, dec = self.enc, self.dec
        opt_enc, opt_dec = self.opt_enc, self.opt_dec
        crit_recon = self.crit_recon

        x, classes, ref = self.data_provider.get_sample()
        x = x.cuda(gpu_id)

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
        z_ref, z_struct = enc(x, classes_onehot)

        total_kld_ref, _, mean_kld_ref = bvae.kl_divergence(z_ref[0], z_ref[1])
        total_kld_struct, _, mean_kld_struct = bvae.kl_divergence(
            z_struct[0], z_struct[1]
        )

        if self.kld_avg:
            kld_ref = mean_kld_ref
            kld_struct = mean_kld_struct
        else:
            kld_ref = total_kld_ref
            kld_struct = total_kld_struct

        kld = kld_ref + kld_struct

        kld_loss_ref = kld_ref.item()
        kld_loss_struct = kld_struct.item()

        zLatent = z_struct[0].data.cpu()

        zAll = [z_ref, z_struct]
        for i in range(len(zAll)):
            zAll[i] = bvae.reparameterize(zAll[i][0], zAll[i][1])

        xHat = dec([classes_onehot] + zAll)

        # Update the image reconstruction
        recon_loss = crit_recon(
            xHat, x
        )  # + crit_recon(xHat2[:,dec.ch_target], x[:,dec.ch_target])

        if self.objective == "H":
            beta_vae_loss = recon_loss + self.beta * kld
        elif self.objective == "B":
            C = torch.clamp(
                torch.Tensor(
                    [self.c_max / self.c_iters_max * len(self.logger)]
                ).type_as(x),
                0,
                self.c_max,
            )
            beta_vae_loss = recon_loss + self.gamma * (kld - C).abs()

        with amp.scale_loss(beta_vae_loss, [opt_enc, opt_dec]) as scaled_loss:
            scaled_loss.backward()

        recon_loss = recon_loss.item()

        opt_enc.step()
        opt_dec.step()

        errors = [recon_loss, kld_loss_ref, kld_loss_struct]

        return errors, zLatent

    def save(self, save_dir):
        #         for saving and loading see:
        #         https://discuss.pytorch.org/t/how-to-save-load-torch-models/718

        import shutil

        gpu_id = self.gpu_ids[0]

        n_iters = self.get_current_iter()

        #         embeddings = np.concatenate(self.zAll, 0)
        #         pickle.dump(embeddings, open("{0}/embedding.pth".format(save_dir), "wb"))
        #         pickle.dump(
        #             embeddings, open("{0}/embedding_{1}.pth".format(save_dir, n_iters), "wb")
        #         )

        enc_save_path_tmp = "{0}/enc.pth".format(save_dir)
        enc_save_path_final = "{0}/enc_{1}.pth".format(save_dir, n_iters)
        dec_save_path_tmp = "{0}/dec.pth".format(save_dir)
        dec_save_path_final = "{0}/dec_{1}.pth".format(save_dir, n_iters)

        model_utils.save_state(self.enc, self.opt_enc, enc_save_path_tmp, gpu_id)
        shutil.copyfile(enc_save_path_tmp, enc_save_path_final)

        model_utils.save_state(self.dec, self.opt_dec, dec_save_path_tmp, gpu_id)
        shutil.copyfile(dec_save_path_tmp, dec_save_path_final)

        pickle.dump(self.logger, open("{0}/logger.pkl".format(save_dir), "wb"))

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

        with torch.no_grad():
            z = enc(x, classes_onehot)
            for i in range(len(z)):
                z[i] = z[i][0]
            xHat = dec([classes_onehot] + z)

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

        x = data_provider.get_images(test_inds, "test").cuda(gpu_id)
        with torch.no_grad():
            z = enc(x, classes_onehot)
            for i in range(len(z)):
                z[i] = z[i][0]

            xHat = dec([classes_onehot] + z)

        for z_sub in z:
            z_sub.normal_()

        with torch.no_grad():
            xHat_z = dec([classes_onehot] + z)

        imgX = tensor2img(x.data.cpu())
        imgXHat = tensor2img(xHat.data.cpu())
        imgXHat_z = tensor2img(xHat_z.data.cpu())
        imgTestOut = np.concatenate((imgX, imgXHat, imgXHat_z), 0)

        imgOut = np.concatenate((imgTrainOut, imgTestOut))

        scipy.misc.imsave(
            "{0}/progress_{1}.png".format(self.save_dir, int(epoch - 1)), imgOut
        )

        enc.train(True)
        dec.train(True)

        embeddings = np.concatenate(self.zAll, 0)

        #         pickle.dump(embeddings, open("{0}/embedding.pth".format(self.save_dir), "wb"))
        #         pickle.dump(
        #             embeddings,
        #             open(
        #                 "{0}/embedding_{1}.pth".format(self.save_dir, self.get_current_iter()),
        #                 "wb",
        #             ),
        #         )

        pickle.dump(self.logger, open("{0}/logger_tmp.pkl".format(self.save_dir), "wb"))

        # History
        plots.history(self.logger, "{0}/history.png".format(self.save_dir))

        # Short History
        plots.short_history(self.logger, "{0}/history_short.png".format(self.save_dir))

        # Embedding figure
        plots.embeddings(embeddings, "{0}/embedding.png".format(self.save_dir))

        xHat = None
        x = None
