import numpy as np
import pickle
import time

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
    ):

        self.data_provider = data_provider
        self.n_epochs = n_epochs

        self.gpu_ids = gpu_ids

        self.save_dir = save_dir

        self.save_state_iter = save_state_iter
        self.save_progress_iter = save_progress_iter

        self.iters_per_epoch = np.ceil(len(data_provider) / data_provider.batch_size)

        self.zAll = list()

    def iteration(self):
        raise NotImplementedError

    def get_current_iter(self):
        return len(self.logger)

    def get_current_epoch(self, iteration=-1):

        if iteration == -1:
            iteration = self.get_current_iter()

        return np.floor(iteration / self.iters_per_epoch)

    def load(self):
        # This is where we load the model
        raise NotImplementedError

    def save(self, save_dir):
        # This is where we save the model
        raise NotImplementedError

    def maybe_save(self):

        epoch = self.get_current_epoch(self.get_current_iter() - 1)
        epoch_next = self.get_current_epoch(self.get_current_iter())

        saved = False
        if epoch != epoch_next:
            # save the logger every epoch
            pickle.dump(
                self.logger, open("{0}/logger_tmp.pkl".format(self.save_dir), "wb")
            )

            if (epoch_next % self.save_progress_iter) == 0:
                print("saving progress")
                self.save_progress()

            if (epoch_next % self.save_state_iter) == 0:
                print("saving state")
                self.save(self.save_dir)

            saved = True

        return saved

    def save_progress(self):
        raise NotImplementedError

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
            self.zAll.append(zLatent.data.cpu().detach().numpy())

            if self.maybe_save():
                self.zAll = list()
