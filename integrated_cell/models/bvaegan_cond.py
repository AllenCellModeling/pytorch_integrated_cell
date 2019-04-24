import torch
import numpy as np
from .. import utils
from . import bvae
from . import bvaegan
from .. import SimpleLogger

import scipy

from integrated_cell.model_utils import tensor2img
from integrated_cell.utils import plots as plots

import os
import pickle


class Model(bvaegan.Model):
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
        channels_ref=[0, 2],
        channels_target=[1],
    ):

        super(Model, self).__init__(
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
            crit_z_class=crit_z_class,
            crit_z_ref=crit_z_ref,
            save_state_iter=save_state_iter,
            save_progress_iter=save_progress_iter,
            beta=beta,
            c_max=c_max,
            c_iters_max=c_iters_max,
            gamma=gamma,
            objective=objective,
            lambda_decD_loss=lambda_decD_loss,
            lambda_ref_loss=lambda_ref_loss,
            lambda_class_loss=lambda_class_loss,
            provide_decoder_vars=provide_decoder_vars,
        )

        self.channels_ref = channels_ref
        self.channels_target = channels_target

        logger_path = "{}/logger.pkl".format(save_dir)
        if os.path.exists(logger_path):
            self.logger = pickle.load(open(logger_path, "rb"))
        else:
            columns = ("epoch", "iter", "reconLoss")
            print_str = "[%d][%d] reconLoss: %.6f"

            columns += ("kldLoss", "minimaxDecDLoss", "decDLoss", "time")
            print_str += " kld: %.6f mmDecD: %.6f decD: %.6f time: %.2f"
            self.logger = SimpleLogger(columns, print_str)

    def iteration(self):
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
        x, classes, _ = self.data_provider.get_sample()

        x = x.cuda(gpu_id)
        xHat_full = x.clone()

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

        for p in decD.parameters():
            p.requires_grad = True

        for p in enc.parameters():
            p.requires_grad = False

        for p in dec.parameters():
            p.requires_grad = False

        zAll = enc(x[:, self.channels_ref], x[:, self.channels_target])

        zAll[-1] = bvae.reparameterize(zAll[-1][0], zAll[-1][1])
        for i in range(len(zAll)):
            zAll[i].detach_()

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
        xHat_full[:, self.channels_target] = xHat
        yHat_xFake = decD(xHat_full)
        errDecD_fake = crit_decD(yHat_xFake, y_xFake)

        # train with fake, sampled and decoded
        for i in range(len(zAll)):
            zAll[i].normal_()

        shuffle_inds = torch.randperm(x.shape[0])
        xHat_full[:, self.channels_target] = dec([classes_onehot[shuffle_inds]] + zAll)
        yHat_xFake2 = decD(xHat_full)
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
        zAll = enc(x[:, self.channels_ref], x[:, self.channels_target])

        total_kld, dimension_wise_kld, mean_kld = bvae.kl_divergence(
            zAll[-1][0], zAll[-1][1]
        )

        zLatent = zAll[-1][0].data.cpu()
        zAll[-1] = bvae.reparameterize(zAll[-1][0], zAll[-1][1])

        xHat = dec([classes_onehot] + zAll)

        # Update the image reconstruction
        recon_loss = crit_recon(xHat, x[:, self.channels_target])

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

        beta_vae_loss.backward(retain_graph=True)
        kld_loss = total_kld.item()

        recon_loss = recon_loss.item()

        opt_enc.step()

        for p in enc.parameters():
            p.requires_grad = False

        # update wrt decD(dec(enc(X)))
        xHat_full[:, self.channels_target] = xHat

        yHat_xFake = decD(xHat_full)
        minimaxDecDLoss = crit_decD(yHat_xFake, y_xReal)

        # update wrt decD(dec(Z))
        for i in range(len(zAll)):
            zAll[i].normal_()

        shuffle_inds = torch.randperm(x.shape[0])
        xHat_full[:, self.channels_target] = dec([classes_onehot[shuffle_inds]] + zAll)

        yHat_xFake2 = decD(xHat_full)
        minimaxDecDLoss2 = crit_decD(yHat_xFake2, y_xReal[shuffle_inds])

        minimaxDecLoss = (minimaxDecDLoss + minimaxDecDLoss2) / 2
        minimaxDecLoss.mul(self.lambda_decD_loss).backward()
        minimaxDecLoss = minimaxDecLoss.item()

        opt_dec.step()

        errors = [recon_loss, kld_loss, minimaxDecLoss, decDLoss]

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

        x, classes, _ = data_provider.get_sample("train", train_inds)
        x = x.cuda(gpu_id)

        classes = classes.type_as(x).long()
        classes_onehot = utils.index_to_onehot(
            classes, self.data_provider.get_n_classes()
        )

        with torch.no_grad():
            z = enc(x[:, self.channels_ref], x[:, self.channels_target])
            z[-1] = z[-1][0]

            x_train = dec([classes_onehot] + z)

        xHat = torch.zeros(x.shape).type_as(x)
        xHat[:, self.channels_target] = x_train

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

        x, classes, _ = data_provider.get_sample("test", test_inds)
        x = x.cuda(gpu_id)
        classes = classes.type_as(x).long()

        x = data_provider.get_images(test_inds, "test").cuda(gpu_id)
        with torch.no_grad():
            z = enc(x[:, self.channels_ref], x[:, self.channels_target])
            z[-1] = z[-1][0]

            x_test = dec([classes_onehot] + z)

        xHat = torch.zeros(x.shape).type_as(x)
        xHat[:, self.channels_target] = x_test

        # sampled location space
        z[-1].normal_()
        with torch.no_grad():
            x_z = dec([classes_onehot] + z)

        xHat_z = x.clone()
        xHat_z[:, self.channels_target] = x_z

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

        # pdb.set_trace()
        # zAll = torch.cat(zAll,0).cpu().numpy()

        embedding = torch.cat(self.zAll, 0).cpu().numpy()

        pickle.dump(
            embedding, open("{0}/embedding_tmp.pkl".format(self.save_dir), "wb")
        )
        pickle.dump(self.logger, open("{0}/logger_tmp.pkl".format(self.save_dir), "wb"))

        # History
        plots.history(self.logger, "{0}/history.png".format(self.save_dir))

        # Short History
        plots.short_history(self.logger, "{0}/history_short.png".format(self.save_dir))

        # Embedding figure
        plots.embeddings(embedding, "{0}/embedding.png".format(self.save_dir))

        xHat = None
        x = None
