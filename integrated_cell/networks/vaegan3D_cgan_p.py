import torch

from . import vaegan3D_cgan

# parallel wrapper for vaegan3D_cgan,
# should allow for models on different gpus with different numbers of gpus


class Enc(vaegan3D_cgan.Enc):
    def __init__(self, output_device=None, **kwargs):
        super(Enc, self).__init__(**kwargs)

        self.output_device = output_device

    def forward(self, x, x_class):

        if len(self.gpu_ids) > 0:
            x = x.cuda(self.gpu_ids[0])
            x_class = x_class.cuda(self.gpu_ids[0])

        if len(self.gpu_ids) > 1:
            return torch.nn.parallel.data_parallel(
                super().forward,
                [x, x_class],
                device_ids=self.gpu_ids,
                output_device=self.output_device,
            )
        else:
            z = super().forward(x, x_class)

            if (len(self.gpu_ids) > 0) and (self.output_device is not None):
                for i in len(z):
                    for zSub in z[i]:
                        z[i] = zSub.cuda(self.output_device)

            return z


class Dec(vaegan3D_cgan.Dec):
    def __init__(self, output_device=None, **kwargs):
        super(Dec, self).__init__(**kwargs)

        self.output_device = output_device

    def forward(self, x_in):
        z_class, z_ref, z_target = x_in

        if len(self.gpu_ids) > 0:
            z_class = z_class.cuda(self.gpu_ids[0])
            z_ref = z_ref.cuda(self.gpu_ids[0])
            z_target = z_target.cuda(self.gpu_ids[0])

        if len(self.gpu_ids) > 1:
            return torch.nn.parallel.data_parallel(
                super().forward,
                [[z_class, z_ref, z_target]],
                device_ids=self.gpu_ids,
                output_device=self.output_device,
            )
        else:
            x_hat = super().forward([z_class, z_ref, z_target])

            if (len(self.gpu_ids) > 0) and (self.output_device is not None):
                x_hat = x_hat.cuda(self.output_device)

            return x_hat


class DecD(vaegan3D_cgan.DecD):
    def __init__(self, output_device=None, **kwargs):
        super(DecD, self).__init__(**kwargs)

        self.output_device = output_device

    def forward(self, x_in, y_in=None):
        if len(self.gpu_ids) > 0:
            x_in = x_in.cuda(self.gpu_ids[0])

        if len(self.gpu_ids) > 1:
            return torch.nn.parallel.data_parallel(
                super().forward, [x_in, y_in], device_ids=self.gpu_ids
            )
        else:
            out = super().forward(x_in, y_in)

            if (len(self.gpu_ids) > 0) and (self.output_device is not None):
                out = out.cuda(self.output_device)

            return out
