import torch
import os

# from .utils import plots
import matplotlib as mpl

mpl.use("Agg")  # noqa

import warnings


def init_opts(opt, opt_default):
    vars_default = vars(opt_default)
    for var in vars_default:
        if not hasattr(opt, var):
            setattr(opt, var, getattr(opt_default, var))
    return opt


def set_gpu_recursive(var, gpu_id):
    for key in var:
        if isinstance(var[key], dict):
            var[key] = set_gpu_recursive(var[key], gpu_id)
        else:
            try:
                if gpu_id != -1:
                    var[key] = var[key].cuda(gpu_id)
                else:
                    var[key] = var[key].cpu()
            except AttributeError:
                pass
    return var


def sampleUniform(batsize, nlatentdim):
    return torch.Tensor(batsize, nlatentdim).uniform_(-1, 1)


def sampleGaussian(batsize, nlatentdim):
    return torch.Tensor(batsize, nlatentdim).normal_()


def tensor2img(img):
    warnings.warn(
        "integrated_cell.model_utils.tensor2img is depricated. Please use integrated_cell.utils.plots.tensor2im instead."
    )
    # return plots.tensor2im(img)


def load_embeddings(embeddings_path, enc=None, dp=None):

    if os.path.exists(embeddings_path):

        embeddings = torch.load(embeddings_path)
    else:
        embeddings = get_latent_embeddings(enc, dp)
        torch.save(embeddings, embeddings_path)

    return embeddings


def get_latent_embeddings(enc, dp):
    enc.eval()
    gpu_id = enc.gpu_ids[0]

    modes = ("test", "train")

    embedding = dict()

    for mode in modes:
        ndat = dp.get_n_dat(mode)
        embeddings = torch.zeros(ndat, enc.n_latent_dim)

        inds = list(range(0, ndat))
        data_iter = [
            inds[i : i + dp.batch_size]  # noqa
            for i in range(0, len(inds), dp.batch_size)
        ]

        for i in range(0, len(data_iter)):
            print(str(i) + "/" + str(len(data_iter)))
            x = dp.get_images(data_iter[i], mode).cuda(gpu_id)

            with torch.no_grad():
                zAll = enc(x)

            embeddings.index_copy_(
                0, torch.LongTensor(data_iter[i]), zAll[-1].data[:].cpu()
            )

        embedding[mode] = embeddings

    return embedding


def load_state(model, optimizer, path, gpu_id):
    # device = torch.device('cpu')

    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # model.cuda(gpu_id)

    # optimizer.state = set_gpu_recursive(optimizer.state, gpu_id)


def save_state(model, optimizer, path, gpu_id):

    # model = model.cpu()
    # optimizer.state = set_gpu_recursive(optimizer.state, -1)

    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(checkpoint, path)

    # model = model.cuda(gpu_id)
    # optimizer.state = set_gpu_recursive(optimizer.state, gpu_id)
