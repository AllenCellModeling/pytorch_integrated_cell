import torch
import numpy as np
import scipy.misc
import pickle
import time

from integrated_cell.model_utils import tensor2img
from integrated_cell.utils import plots as plots


# This is the base class for trainers


class Model(object):
    def __init__(
        self,
        data_provider,
        n_epochs,
        gpu_ids,
        save_dir,
        save_state_iter=1,
        save_progress_iter=1,
        **kwargs
    ):

        self.__dict__.update(kwargs)

        self.data_provider = data_provider
        self.n_epochs = n_epochs

        self.gpu_ids = gpu_ids

        self.save_dir = save_dir

        self.save_state_iter = save_state_iter
        self.save_progress_iter = save_progress_iter

        self.iters_per_epoch = np.ceil(len(data_provider) / data_provider.batch_size)

        self.zAll = list()

    def get_current_iter(self):
        return len(self.logger)

    def get_current_epoch(self, iteration=-1):

        if iteration == -1:
            iteration = self.get_current_iter()

        return np.floor(iteration / self.iters_per_epoch)

    def load(self):
        raise NotImplementedError

    def save(self, save_dir):
        raise NotImplementedError

    def maybe_save(self):

        epoch = self.get_current_epoch(self.get_current_iter() - 1)
        epoch_next = self.get_current_epoch(self.get_current_iter())

        saved = False
        if epoch != epoch_next and (
            (epoch_next % self.save_state_iter) == 0
            or (epoch_next % self.save_state_iter) == 0
        ):
            if (epoch_next % self.save_progress_iter) == 0:
                print("saving progress")
                self.save_progress()

            if (epoch_next % self.save_progress_iter) == 0:
                print("saving state")
                self.save(self.save_dir)

            saved = True

        return saved

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

        x = data_provider.get_images(train_inds, "train").cuda(gpu_id)

        with torch.no_grad():
            xHat = dec(enc(x))

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

        x = data_provider.get_images(test_inds, "test").cuda(gpu_id)
        with torch.no_grad():
            xHat = dec(enc(x))

        z = list()
        if enc.n_classes > 0:
            class_var = torch.Tensor(
                data_provider.get_classes(test_inds, "test", "one_hot").float()
            ).cuda(gpu_id)
            class_var = (class_var - 1) * 25
            z.append(class_var)

        if enc.n_ref > 0:
            ref_var = (
                torch.Tensor(data_provider.get_n_classes(), enc.n_ref)
                .normal_(0, 1)
                .cuda(gpu_id)
            )
            z.append(ref_var)

        loc_var = (
            torch.Tensor(data_provider.get_n_classes(), enc.n_latent_dim)
            .normal_(0, 1)
            .cuda(gpu_id)
        )
        z.append(loc_var)

        with torch.no_grad():
            x_z = dec(z)

        imgX = tensor2img(x.data.cpu())
        imgXHat = tensor2img(xHat.data.cpu())
        imgX_z = tensor2img(x_z.data.cpu())
        imgTestOut = np.concatenate((imgX, imgXHat, imgX_z), 0)

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

    def train(self):
        start_iter = self.get_current_iter()

        for this_iter in range(
            int(start_iter), int(np.ceil(self.iters_per_epoch) * self.n_epochs)
        ):

            start = time.time()

            errors, zLatent = self.iteration()

            stop = time.time()
            deltaT = stop - start

            self.logger.add(
                [self.get_current_epoch(), self.get_current_iter()] + errors + [deltaT]
            )
            self.zAll.append(zLatent.data.cpu())

            if self.maybe_save():
                self.zAll = list()


#     def setup_decoder_vars(self, z, classes, ref):
#         if self.provide_decoder_vars:
#             c = 0
#             if self.n_classes > 0:
#                 z[c] = torch.log(utils.index_to_onehot(classes, self.data_provider.get_n_classes()) + 1E-8)
#                 c += 1

#             if self.n_ref > 0:
#                 z[c] = ref
#                 c += 1

#         return z
