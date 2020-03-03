import torch
import torch.nn as nn
from .. import utils


class Autoencoder(nn.Module):
    def __init__(self, ref_enc, ref_dec, target_enc, target_dec):
        super(Autoencoder, self).__init__()

        self.ref_enc = ref_enc
        self.ref_dec = ref_dec
        self.ref_n_latent_dim = ref_enc.n_latent_dim

        self.target_enc = target_enc
        self.target_dec = target_dec
        self.target_n_latent_dim = target_enc.n_latent_dim

    def encode(self, target, ref, labels):
        batch_size = labels.shape[0]

        if ref is None:
            # generate ref image
            self.z_ref = (
                torch.zeros([batch_size, self.ref_n_latent_dim])
                .type_as(labels)
                .normal_()
            )

        else:
            self.z_ref = self.ref_enc(ref)

        if target is None:

            self.z_target = (
                torch.zeros([batch_size, self.target_n_latent_dim])
                .type_as(labels)
                .normal_()
            )

        else:
            self.z_target = self.target_enc(target, ref, labels)

        return self.z_ref, self.z_target

    def forward(self, target, ref, labels):

        batch_size = labels.shape[0]

        if ref is None:
            # generate ref image
            self.z_ref = (
                torch.zeros([batch_size, self.ref_n_latent_dim])
                .type_as(labels)
                .normal_()
            )

            ref_hat = self.ref_dec(self.z_ref)
            ref = ref_hat
        else:

            self.z_ref = self.ref_enc(ref)
            self.z_ref_sampled = utils.reparameterize(self.z_ref[0], self.z_ref[1])

            ref_hat = self.ref_dec(self.z_ref_sampled)

        if target is None:

            self.z_target = (
                torch.zeros([batch_size, self.target_n_latent_dim])
                .type_as(labels)
                .normal_()
            )

            target_hat = self.target_dec(self.z_target, ref, labels)
        else:
            self.z_target = self.target_enc(target, ref, labels)
            self.z_target_sampled = utils.reparameterize(
                self.z_target[0], self.z_target[1]
            )
            target_hat = self.target_dec(self.z_target_sampled, ref, labels)

        return target_hat, ref_hat
