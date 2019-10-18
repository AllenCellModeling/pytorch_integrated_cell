import torch
from .. import utils
from . import cbvae
from .bvae import reparameterize, kl_divergence

from apex import amp


class Model(cbvae.Model):
    def __init__(self, opt_level="O1", **kwargs):

        super(Model, self).__init__(**kwargs)

        is_data_parallel = False
        # do some hack if set up for DataParallel
        if (
            str(self.enc.__class__)
            == "<class 'torch.nn.parallel.data_parallel.DataParallel'>"
        ):
            is_data_parallel = True
            device_ids_enc = self.enc.device_ids
            device_ids_dec = self.dec.device_ids

            self.enc = self.enc.module
            self.dec = self.dec.module

        [self.enc, self.dec], [self.opt_enc, self.opt_dec] = amp.initialize(
            [self.enc, self.dec], [self.opt_enc, self.opt_dec], opt_level=opt_level
        )

        if is_data_parallel:
            self.enc = torch.nn.DataParallel(self.enc, device_ids_enc)
            self.dec = torch.nn.DataParallel(self.dec, device_ids_dec)

    def iteration(self):

        torch.cuda.empty_cache()

        gpu_id = self.gpu_ids[0]

        enc, dec = self.enc, self.dec
        opt_enc, opt_dec = self.opt_enc, self.opt_dec
        crit_recon = self.crit_recon

        # do this just incase anything upstream changes these values
        enc.train(True)
        dec.train(True)

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

        kld_ref, _, _ = kl_divergence(z_ref[0], z_ref[1])
        kld_struct, _, _ = kl_divergence(z_struct[0], z_struct[1])

        kld = kld_ref + kld_struct

        kld_loss_ref = kld_ref.item()
        kld_loss_struct = kld_struct.item()

        zLatent = z_struct[0].data.cpu()

        zAll = [z_ref, z_struct]
        for i in range(len(zAll)):
            zAll[i] = reparameterize(zAll[i][0], zAll[i][1])

        xHat = dec([classes_onehot] + zAll)

        # Update the image reconstruction
        recon_loss = crit_recon(xHat, x)

        if self.objective == "H":
            beta_vae_loss = recon_loss + self.beta * kld
        elif self.objective == "H_eps":
            beta_vae_loss = (
                recon_loss
                + torch.abs((self.beta * kld_ref) - x.shape[0] * 0.1)
                + torch.abs((self.beta * kld_struct) - x.shape[0] * 0.1)
            )
        elif self.objective == "B":
            C = torch.clamp(
                torch.Tensor(
                    [self.c_max / self.c_iters_max * len(self.logger)]
                ).type_as(x),
                0,
                self.c_max,
            )
            beta_vae_loss = recon_loss + self.gamma * (kld - C).abs()

        elif self.objective == "B_eps":
            C = torch.clamp(
                torch.Tensor(
                    [self.c_max / self.c_iters_max * len(self.logger)]
                ).type_as(x),
                0,
                self.c_max,
            )
            beta_vae_loss = recon_loss + self.gamma * (kld - C).abs()

        elif self.objective == "A":
            # warmup mode
            beta_mult = self.beta_start + self.beta_step * self.get_current_iter()
            if beta_mult > self.beta_max:
                beta_mult = self.beta_max

            if beta_mult < self.beta_min:
                beta_mult = self.beta_min

            beta_vae_loss = recon_loss + beta_mult * kld

        with amp.scale_loss(beta_vae_loss, [opt_enc, opt_dec]) as scaled_loss:
            scaled_loss.backward()

        recon_loss = recon_loss.item()

        opt_enc.step()
        opt_dec.step()

        errors = [recon_loss, kld_loss_ref, kld_loss_struct]

        return errors, zLatent
