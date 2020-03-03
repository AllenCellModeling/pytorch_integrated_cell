import torch
from integrated_cell.data_providers.DataProvider import (
    DataProvider as ParentDataProvider,
)  # ugh im sorry

from .. import utils


class DataProvider(ParentDataProvider):
    # Same as DataProvider but zeros out channels indicated by the variable 'masked_channels'
    def __init__(self, masked_channels=[], **kwargs):

        super().__init__(**kwargs)

        self.masked_channels = masked_channels

    def get_sample(self, train_or_test="train", inds=None):
        # returns
        # x         is b by c by y by x
        # x_class   is b by c by #classes
        # graph     is b by c by c - a random dag over channels

        x, classes, _ = super().get_sample(train_or_test=train_or_test, inds=inds)

        n_b = x.shape[0]
        n_c = x.shape[1]

        classes = classes.type_as(x).long()
        classes = utils.index_to_onehot(classes, self.get_n_classes())

        classes_mem = torch.ones(n_b) * self.get_n_classes() - 2
        classes_mem = utils.index_to_onehot(classes_mem, self.get_n_classes())

        classes_dna = torch.ones(n_b) * self.get_n_classes() - 1
        classes_dna = utils.index_to_onehot(classes_dna, self.get_n_classes())

        classes = [torch.unsqueeze(c, 1) for c in [classes_mem, classes, classes_dna]]
        classes = torch.cat(classes, 1)

        graph = torch.zeros(x.shape[0], x.shape[1], x.shape[1])

        graphs = list()
        for i in range(n_b):
            graph = (
                torch.ones([n_c, n_c]).triu(1) * torch.zeros([n_c, n_c]).bernoulli_()
            )
            rand_inds = torch.randperm(n_c)

            graph = graph[:, rand_inds][rand_inds, :]
            graphs.append(graph)

        graph = torch.stack(graphs, 0).long()

        return x, classes, graph

    def get_n_classes(self):
        return super().get_n_classes() + 2
