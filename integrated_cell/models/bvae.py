import torch
from .. import model_utils
from .. import utils
from .. import SimpleLogger
from . import base_model
import pickle
import os

# This is the trainer for the Beta-VAE


def reparameterize(mu, log_var, add_noise=True):
    if add_noise:
        std = log_var.div(2).exp()
        eps = torch.randn_like(std)
        out = eps.mul(std).add_(mu)
    else:
        out = mu

    return out


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

            if crit_z_class is not None:
                columns += ("class_loss",)
                print_str += " class_loss: %.6f"

            if crit_z_ref is not None:
                columns += ("ref_loss",)
                print_str += " ref_loss: %.6f"

            columns += ("kldLoss", "time")
            print_str += " kld: %.6f time: %.2f"
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

        enc.train(True)
        dec.train(True)

        opt_enc.zero_grad()
        opt_dec.zero_grad()

        # Forward passes
        zAll = enc(x)

        c = 0
        # Update the class discriminator
        if crit_z_class is not None:
            class_loss = crit_z_class(zAll[c], classes)
            class_loss.backward(retain_graph=True)
            class_loss = class_loss.item()

            if self.provide_decoder_vars:
                zAll[c] = torch.log(
                    utils.index_to_onehot(classes, self.data_provider.get_n_classes())
                    + 1e-8
                )

            c += 1

        # Update the reference shape discriminator
        if crit_z_ref is not None:
            ref_loss = crit_z_ref(zAll[c], ref)
            ref_loss.backward(retain_graph=True)
            ref_loss = ref_loss.item()

            if self.provide_decoder_vars:
                zAll[c] = ref

            c += 1

        total_kld, dimension_wise_kld, mean_kld = kl_divergence(zAll[c][0], zAll[c][1])

        if self.kld_avg:
            kld = mean_kld
        else:
            kld = total_kld

        zLatent = zAll[c][0].data.cpu()

        zAll[c] = reparameterize(zAll[c][0], zAll[c][1])

        xHat = dec(zAll)

        # Update the image reconstruction
        recon_loss = crit_recon(xHat, x)

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

        beta_vae_loss.backward()

        kld_loss = total_kld.item()
        recon_loss = recon_loss.item()

        errors = [recon_loss]
        if enc.n_classes > 0:
            errors += [class_loss]

        if enc.n_ref > 0:
            errors += [ref_loss]

        errors += [kld_loss]
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
