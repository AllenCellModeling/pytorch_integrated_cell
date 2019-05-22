import torch
import numpy as np
from .. import model_utils
from .. import utils
from . import bvae
from . import base_model
from .. import SimpleLogger

import os
import pickle


class Model(base_model.Model):
    def __init__(
        self,
        enc,
        dec,
        decD,
        opt_enc,
        opt_dec,
        opt_decD,
        n_epochs,
        gpu_ids,
        save_dir,
        data_provider,
        crit_recon,
        crit_decD,
        crit_z_class=None,
        crit_z_ref=None,
        save_state_iter=1,
        save_progress_iter=1,
        beta=1,
        c_max=25,
        c_iters_max=1.2e5,
        gamma=1000,
        objective="H",
        lambda_decD_loss=1e-4,
        lambda_ref_loss=1,
        lambda_class_loss=1,
        provide_decoder_vars=False,
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

        self.enc = enc
        self.dec = dec
        self.decD = decD

        self.opt_enc = opt_enc
        self.opt_dec = opt_dec
        self.opt_decD = opt_decD

        self.crit_recon = crit_recon
        self.crit_z_class = crit_z_class
        self.crit_z_ref = crit_z_ref
        self.crit_decD = crit_decD

        if objective == "H":
            self.beta = beta
        elif objective == "B":
            self.c_max = c_max
            self.gamma = gamma
            self.c_iters_max = c_iters_max

        self.provide_decoder_vars = provide_decoder_vars
        self.objective = objective

        self.lambda_decD_loss = lambda_decD_loss
        self.lambda_ref_loss = lambda_ref_loss
        self.lambda_class_loss = lambda_class_loss

        logger_path = "{}/logger.pkl".format(save_dir)
        if os.path.exists(logger_path):
            self.logger = pickle.load(open(logger_path, "rb"))
        else:
            columns = ("epoch", "iter", "reconLoss")
            print_str = "[%d][%d] reconLoss: %.6f"

            if crit_z_class is not None:
                columns += ("class_loss",)
                print_str += " class_loss: %.6f"

            if crit_z_ref is not None:
                columns += ("ref_loss",)
                print_str += " ref_loss: %.6f"

            columns += ("kldLoss", "minimaxDecDLoss", "decDLoss", "time")
            print_str += " kld: %.6f mmDecD: %.6f decD: %.6f time: %.2f"
            self.logger = SimpleLogger(columns, print_str)

    def iteration(self):
        gpu_id = self.gpu_ids[0]

        enc, dec, decD = self.enc, self.dec, self.decD
        opt_enc, opt_dec, opt_decD = self.opt_enc, self.opt_dec, self.opt_decD
        crit_recon, crit_z_class, crit_z_ref, crit_decD = (
            self.crit_recon,
            self.crit_z_class,
            self.crit_z_ref,
            self.crit_decD,
        )

        # do this just incase anything upstream changes these values
        enc.train(True)
        dec.train(True)
        decD.train(True)

        # update the discriminator
        # maximize log(AdvZ(z)) + log(1 - AdvZ(Enc(x)))
        x, classes, ref = self.data_provider.get_sample()

        x = x.cuda(gpu_id)

        if crit_z_class is None:
            y_xReal = torch.ones(x.shape[0], 1).type_as(x).long()
            y_xFake = torch.zeros(x.shape[0], 1).type_as(x).long()
        else:
            classes = classes.type_as(x).long()

            y_xReal = classes
            y_xFake = (
                torch.zeros(classes.shape)
                .fill_(self.data_provider.get_n_classes())
                .type_as(x)
                .long()
            )

        if crit_z_ref is not None:
            ref = ref.type_as(x)

        for p in decD.parameters():
            p.requires_grad = True

        for p in enc.parameters():
            p.requires_grad = False

        for p in dec.parameters():
            p.requires_grad = False

        zAll = enc(x)

        for i in range(len(zAll) - 1):
            zAll[i].detach_()

        for var in zAll[-1]:
            var.detach_()

        zAll[-1] = bvae.reparameterize(zAll[-1][0], zAll[-1][1])

        xHat = dec(zAll)

        zReal = torch.zeros(zAll[-1].shape).normal_().type_as(x)

        opt_enc.zero_grad()
        opt_dec.zero_grad()
        opt_decD.zero_grad()

        ##############
        # Train decD
        ##############

        yHat_xReal = decD(x)

        # train with real
        errDecD_real = crit_decD(yHat_xReal, y_xReal)

        # train with fake, reconstructed
        yHat_xFake = decD(xHat)

        errDecD_fake = crit_decD(yHat_xFake, y_xFake)

        # train with fake, sampled and decoded
        zAll[-1] = zReal

        yHat_xFake2 = decD(dec(zAll))
        errDecD_fake2 = crit_decD(yHat_xFake2, y_xFake)

        decDLoss = (errDecD_real + (errDecD_fake + errDecD_fake2) / 2) / 2
        decDLoss.backward(retain_graph=True)
        opt_decD.step()

        decDLoss = decDLoss.item()

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

        # Forward passes
        zAll = enc(x)

        c = 0
        # Update the class discriminator
        loss = torch.zeros(1).type_as(x).float()

        classLoss = torch.zeros(1)
        if crit_z_class is not None:
            classLoss = crit_z_class(zAll[c], classes)
            loss += classLoss.mul(self.lambda_class_loss)
            classLoss_tmp = classLoss.item()

            if self.provide_decoder_vars:
                zAll[c] = torch.log(
                    utils.index_to_onehot(classes, self.data_provider.get_n_classes())
                    + 1e-8
                )

            c += 1

        # Update the reference shape discriminator
        if crit_z_ref is not None:
            refLoss = crit_z_ref(zAll[c], ref)
            loss += refLoss.mul(self.lambda_ref_loss)
            refLoss_tmp = refLoss.item()

            if self.provide_decoder_vars:
                zAll[c] = ref

            c += 1

        total_kld, dimension_wise_kld, mean_kld = bvae.kl_divergence(
            zAll[c][0], zAll[c][1]
        )

        zLatent = zAll[c][0].data.cpu()

        zAll[c] = bvae.reparameterize(zAll[c][0], zAll[c][1])

        xHat = dec(zAll)

        # Update the image reconstruction
        recon_loss = crit_recon(xHat, x)

        if self.objective == "H":
            beta_vae_loss = recon_loss + self.beta * total_kld
        elif self.objective == "B":
            C = torch.clamp(
                torch.Tensor(
                    [self.c_max / self.c_iters_max * len(self.logger)]
                ).type_as(x),
                0,
                self.c_max,
            )
            beta_vae_loss = recon_loss + self.gamma * (total_kld - C).abs()

        loss += beta_vae_loss

        loss.backward(retain_graph=True)

        kld_loss = total_kld.item()
        recon_loss = recon_loss.item()

        opt_enc.step()

        for p in enc.parameters():
            p.requires_grad = False

        # update wrt decD(dec(enc(X)))
        yHat_xFake = decD(xHat)
        minimaxDecDLoss = crit_decD(yHat_xFake, y_xReal)
        yHat_xFake = None

        # update wrt decD(dec(Z))

        c = 0
        # if we have classes, create random classes, generate images of random classes
        if crit_z_class is not None:
            shuffle_inds = np.arange(0, zAll[0].size(0))

            classes_one_hot = (
                torch.log(
                    utils.index_to_onehot(classes, self.data_provider.get_n_classes())
                    + 1e-8
                )
                .type_as(zAll[c].data)
                .cuda(gpu_id)
            )

            np.random.shuffle(shuffle_inds)
            zAll[c] = classes_one_hot[shuffle_inds, :]
            y_xReal = y_xReal[torch.LongTensor(shuffle_inds).cuda(gpu_id)]

            c += 1

        if crit_z_ref is not None:
            zAll[c].data.normal_()

        # sample random positions in the localization space
        zAll[-1].normal_()

        xHat = dec(zAll)

        yHat_xFake2 = decD(xHat)
        minimaxDecDLoss2 = crit_decD(yHat_xFake2, y_xReal)
        yHat_xFake2 = None

        minimaxDecLoss = (minimaxDecDLoss + minimaxDecDLoss2) / 2
        minimaxDecLoss.mul(self.lambda_decD_loss).backward()
        minimaxDecLoss = minimaxDecLoss.item()
        opt_dec.step()

        errors = [recon_loss]
        if crit_z_class is not None:
            errors += [classLoss_tmp]

        if crit_z_ref is not None:
            errors += [refLoss_tmp]

        errors += [kld_loss, minimaxDecLoss, decDLoss]

        return errors, zLatent

    def save(self, save_dir):
        #         for saving and loading see:
        #         https://discuss.pytorch.org/t/how-to-save-load-torch-models/718

        import shutil

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
