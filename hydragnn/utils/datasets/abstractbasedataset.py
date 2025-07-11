from abc import ABC, abstractmethod

import torch

tmp_list = [
    "ani1x",
    "qm7x",
    "mptrj",
    "alexandria",
    "transition1x",
    "oc2020_all",
    "oc2022",
    "omat24",
    "omol25",
]

class AbstractBaseDataset(torch.utils.data.Dataset, ABC):
    """
    HydraGNN's base datasets. This is abstract class.
    """

    def __init__(self):
        super().__init__()
        self.dataset = list()
        self.dataset_name = None
        self.tmp_dict = dict()
        for i, name in enumerate(tmp_list):
            self.tmp_dict[name] = torch.tensor([[i]])

        # self.tmp_dict = {
        #     "ani1x": torch.tensor([[0]]),
        #     "qm7x": torch.tensor([[1]]),
        #     "mptrj": torch.tensor([[2]]),
        #     "alexandria": torch.tensor([[3]]),
        #     "transition1x": torch.tensor([[4]]),
        #     "oc2020_all": torch.tensor([[5]]),
        #     "oc2022": torch.tensor([[6]]),
        #     "omat24": torch.tensor([[7]]),
        #     "omol25": torch.tensor([[8]]),
        # }

    @abstractmethod
    def get(self, idx):
        """
        Return a datasets at idx
        """
        pass

    @abstractmethod
    def len(self):
        """
        Total number of datasets.
        If data is distributed, it should be the global total size.
        """
        pass

    def apply(self, func):
        for data in self.dataset:
            func(data)

    def map(self, func):
        for data in self.dataset:
            yield func(data)

    def __len__(self):
        return self.len()

    def __getitem__(self, idx):
        obj = self.get(idx)
        ## DDStore needs an explicit dimension: 1-by-1
        if hasattr(self, "dataset_name"):
            if self.dataset_name is not None:
                obj.dataset_name = self.tmp_dict[self.dataset_name]
        return obj

    def __iter__(self):
        for idx in range(self.len()):
            yield self.get(idx)
