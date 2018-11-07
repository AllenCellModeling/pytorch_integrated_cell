import torch
import os
import pickle

from integrated_cell import model_utils
from integrated_cell import utils
from integrated_cell.models import bvae
from integrated_cell.models import base_model
from integrated_cell import SimpleLogger


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
        lambda_loss=1e-4,
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
        self.opt_enc = opt_enc
        self.opt_dec = opt_dec

        self.crit_recon = crit_recon
        self.crit_z_class = crit_z_class
        self.crit_z_ref = crit_z_ref

        self.objective = objective
        if objective == "H":
            self.beta = beta
        elif objective == "B":
            self.c_max = c_max
            self.gamma = gamma
            self.c_iters_max = c_iters_max

        self.provide_decoder_vars = provide_decoder_vars

        self.lambda_loss = lambda_loss
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
                columns += ("refLoss",)
                print_str += " refLoss: %.6f"

            columns += ("kldLoss", "selfLoss", "time")
            print_str += " kld: %.6f sLoss: %.6f time: %.2f"
            self.logger = SimpleLogger(columns, print_str)

    def iteration(self):
        gpu_id = self.gpu_ids[0]

        enc, dec = self.enc, self.dec
        opt_enc, opt_dec = self.opt_enc, self.opt_dec
        crit_recon, crit_z_class, crit_z_ref = (
            self.crit_recon,
            self.crit_z_class,
            self.crit_z_ref,
        )

        # do this just incase anything upstream changes these values
        enc.train(True)
        dec.train(True)

        x, classes, ref = self.data_provider.get_sample()

        x = x.cuda(gpu_id)

        if crit_z_ref is not None:
            ref = ref.type_as(x)

        if crit_z_class is not None:
            classes = classes.type_as(x).long()

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
        zAll, activations = enc(x)

        c = 0
        # Update the class discriminator
        if crit_z_class is not None:
            class_loss = crit_z_class(zAll[c], classes) * self.lambda_class_loss
            class_loss.backward(retain_graph=True)
            class_loss = class_loss.item()

            if self.provide_decoder_vars:
                zAll[c] = torch.log(
                    utils.index_to_onehot(classes, enc.n_classes) + 1e-8
                )

            c += 1

        # Update the reference shape discriminator
        if crit_z_ref is not None:
            ref_loss = crit_z_ref(zAll[c], ref) * self.lambda_ref_loss
            ref_loss.backward(retain_graph=True)
            ref_loss = ref_loss.item()

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

        beta_vae_loss.backward(retain_graph=True)

        for p in enc.parameters():
            p.requires_grad = False

        _, activations_hat = enc(xHat)

        self_loss = torch.tensor(0).type_as(x)
        for activation_hat, activation in zip(activations_hat, activations):
            self_loss += crit_recon(activation_hat, activation.detach())

        (self_loss * self.lambda_loss).backward()

        for p in enc.parameters():
            p.requires_grad = True

        opt_enc.step()
        opt_dec.step()

        kld_loss = total_kld.item()
        recon_loss = recon_loss.item()
        self_loss = self_loss.item()

        errors = [recon_loss]
        if enc.n_classes > 0:
            errors += [class_loss]

        if enc.n_ref > 0:
            errors += [ref_loss]

        errors += [kld_loss, self_loss]

        return errors, zLatent

    def save(self, save_dir):
        embeddings = torch.cat(self.zAll, 0).cpu().numpy()

        save_f(
            self.enc,
            self.dec,
            self.opt_enc,
            self.opt_dec,
            self.logger,
            embeddings,
            self.gpu_ids[0],
            save_dir,
        )


def save_f(enc, dec, opt_enc, opt_dec, logger, embeddings, gpu_id, save_dir):
    # for saving and loading see:
    # https://discuss.pytorch.org/t/how-to-save-load-torch-models/718

    import shutil

    n_iters = int(len(logger))

    enc_save_path_tmp = "{0}/enc.pth".format(save_dir)
    enc_save_path_final = "{0}/enc_{1}.pth".format(save_dir, n_iters)
    dec_save_path_tmp = "{0}/dec.pth".format(save_dir)
    dec_save_path_final = "{0}/enc_{1}.pth".format(save_dir, n_iters)

    model_utils.save_state(enc, opt_enc, enc_save_path_tmp, gpu_id)
    shutil.copyfile(enc_save_path_tmp, enc_save_path_final)

    model_utils.save_state(dec, opt_dec, dec_save_path_tmp, gpu_id)
    shutil.copyfile(dec_save_path_tmp, dec_save_path_final)

    pickle.dump(embeddings, open("{0}/embedding.pth".format(save_dir), "wb"))
    pickle.dump(
        embeddings, open("{0}/embedding_{1}.pth".format(save_dir, n_iters), "wb")
    )

    pickle.dump(logger, open("{0}/logger.pkl".format(save_dir), "wb"))
