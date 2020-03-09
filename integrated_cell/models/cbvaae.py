import torch
import numpy as np
import os
import pickle
import shutil

from .. import utils
from . import cbvae2
from .. import SimpleLogger
from .. import model_utils


# conditional beta variational advarsarial auto encoder
#
# This is because sometimes the conditioning results in different latent space embeddings
# and we try to fix that with the advarsary


class Model(cbvae2.Model):
    def __init__(
        self, encD, opt_encD, crit_encD, lambda_encD_loss=1, n_encD_steps=1, **kwargs
    ):

        super(Model, self).__init__(**kwargs)

        self.encD = encD
        self.opt_encD = opt_encD

        self.crit_encD = crit_encD

        self.lambda_encD_loss = lambda_encD_loss
        self.n_encD_steps = n_encD_steps

        logger_path = "{}/logger.pkl".format(self.save_dir)
        if os.path.exists(logger_path):
            self.logger = pickle.load(open(logger_path, "rb"))
        else:
            columns = ("epoch", "iter", "reconLoss")
            print_str = "[%d][%d] reconLoss: %.6f"

            columns += (
                "kldLossRef",
                "kldLossStruct",
                "minimaxencDLoss",
                "encDLoss",
                "time",
            )
            print_str += (
                " kld ref: %.6f kld struct: %.6f mmEncD: %.6f encD: %.6f time: %.2f"
            )
            self.logger = SimpleLogger(columns, print_str)

    def iteration(self):
        gpu_id = self.gpu_ids[0]

        enc, dec, encD = self.enc, self.dec, self.encD
        opt_enc, opt_dec, opt_encD = self.opt_enc, self.opt_dec, self.opt_encD
        crit_recon, crit_encD = self.crit_recon, self.crit_encD

        # do this just incase anything upstream changes these values
        enc.train(True)
        dec.train(True)
        encD.train(True)

        # update the discriminator
        # maximize log(AdvZ(z)) + log(1 - AdvZ(Enc(x)))

        for p in encD.parameters():
            p.requires_grad = True

        for p in enc.parameters():
            p.requires_grad = False

        for p in dec.parameters():
            p.requires_grad = False

        encD_loss = 0
        for step_num in range(self.n_encD_steps):
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

            y_zReal = y_xFake
            y_zFake = y_xReal

            with torch.no_grad():
                zAll = enc(x, classes_onehot)

            zFake = self.reparameterize(zAll[-1][0], zAll[-1][1])
            zReal = torch.zeros(zFake.shape).type_as(zFake).normal_()

            opt_enc.zero_grad()
            opt_dec.zero_grad()
            opt_encD.zero_grad()

            ###############
            # train encD
            ###############

            # train with real
            yHat_zReal = encD(zReal)
            errEncD_real = crit_encD(yHat_zReal, y_zReal)

            # train with fake
            yHat_zFake = encD(zFake)
            errEncD_fake = crit_encD(yHat_zFake, y_zFake)

            encD_loss_tmp = (errEncD_real + errEncD_fake) / 2
            encD_loss_tmp.backward()

            opt_encD.step()

            encD_loss += encD_loss_tmp.item()

        for p in enc.parameters():
            p.requires_grad = True

        for p in dec.parameters():
            p.requires_grad = True

        for p in encD.parameters():
            p.requires_grad = False

        opt_enc.zero_grad()
        opt_dec.zero_grad()
        opt_encD.zero_grad()

        #####################
        # train autoencoder
        #####################

        # Forward passes
        z_ref, z_struct = enc(x, classes_onehot)

        kld_ref = self.kld_loss(z_ref[0], z_ref[1])
        kld_struct = self.kld_loss(z_struct[0], z_struct[1])

        kld_loss = kld_ref + kld_struct

        zAll = [z_ref, z_struct]
        for i in range(len(zAll)):
            zAll[i] = self.reparameterize(zAll[i][0], zAll[i][1])

        xHat = dec([classes_onehot] + zAll)

        recon_loss = crit_recon(xHat, x)

        if self.lambda_encD_loss > 0:
            yHat_zFake = encD(zAll[-1])
            minimax_encD_loss = crit_encD(yHat_zFake, y_xReal)
        else:
            minimax_encD_loss = 0

        beta_vae_loss = (
            self.vae_loss(recon_loss, kld_loss)
            + self.lambda_encD_loss * minimax_encD_loss
        )

        beta_vae_loss.backward()
        opt_enc.step()
        opt_dec.step()

        # Log a bunch of stuff
        recon_loss = recon_loss.item()
        minimax_encD_loss = minimax_encD_loss.item()

        kld_loss_ref = kld_ref.item()
        kld_loss_struct = kld_struct.item()

        zLatent = z_struct[0].data.cpu()

        errors = [
            recon_loss,
            kld_loss_ref,
            kld_loss_struct,
            minimax_encD_loss,
            encD_loss,
        ]

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

        encD_save_path_tmp = "{0}/encD.pth".format(save_dir)
        encD_save_path_final = "{0}/encD_{1}.pth".format(save_dir, n_iters)

        model_utils.save_state(self.enc, self.opt_enc, enc_save_path_tmp, gpu_id)
        shutil.copyfile(enc_save_path_tmp, enc_save_path_final)

        model_utils.save_state(self.dec, self.opt_dec, dec_save_path_tmp, gpu_id)
        shutil.copyfile(dec_save_path_tmp, dec_save_path_final)

        model_utils.save_state(self.encD, self.opt_encD, encD_save_path_tmp, gpu_id)
        shutil.copyfile(encD_save_path_tmp, encD_save_path_final)

        pickle.dump(self.logger, open("{0}/logger.pkl".format(save_dir), "wb"))
        pickle.dump(
            self.logger, open("{0}/logger_{1}.pkl".format(save_dir, n_iters), "wb")
        )
