from abc import ABC, abstractmethod
from fedlab.utils.dataset import functional as F
import numpy as np


class DataPartitioner(ABC):
    """Base class for data partition in federated learning.
    """

    @abstractmethod
    def _perform_partition(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

class BasicPartitioner(DataPartitioner):
    """
    - label-distribution-skew:quantity-based
    - label-distribution-skew:distributed-based (Dirichlet)
    - quantity-skew (Dirichlet)
    - IID

    Args:
        targets:
        num_clients:
        partition:
        dir_alpha:
        major_classes_num:
        verbose:
        seed:
    """
    num_classes = 2

    def __init__(self, targets, num_clients, 
                 num_classes,
                 partition='iid',
                 dir_alpha=None,
                 major_classes_num=1,
                 verbose=True,
                 seed=None):
        self.targets = np.array(targets)  # with shape (num_samples,)
        self.num_samples = self.targets.shape[0]
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.client_dict = dict()
        self.partition = partition
        self.dir_alpha = dir_alpha
        self.verbose = verbose
        # self.rng = np.random.default_rng(seed)  # rng currently not supports randint
        np.random.seed(seed)

        if partition == "noniid-#label":
            # label-distribution-skew:quantity-based
            assert isinstance(major_classes_num, int), f"'major_classes_num' should be integer, " \
                                                       f"not {type(major_classes_num)}."
            assert major_classes_num > 0, f"'major_classes_num' should be positive."
            assert major_classes_num < self.num_classes, f"'major_classes_num' for each client " \
                                                         f"should be less than number of total " \
                                                         f"classes {self.num_classes}."
            self.major_classes_num = major_classes_num
        elif partition in ["noniid-labeldir", "unbalance"]:
            # label-distribution-skew:distributed-based (Dirichlet) and quantity-skew (Dirichlet)
            assert dir_alpha > 0, f"Parameter 'dir_alpha' for Dirichlet distribution should be " \
                                  f"positive."
        elif partition == "iid":
            # IID
            pass
        else:
            raise ValueError(
                f"tabular data partition only supports 'noniid-#label', 'noniid-labeldir', "
                f"'unbalance', 'iid'. {partition} is not supported.")

        self.client_dict = self._perform_partition()
        # get sample number count for each client
        self.client_sample_count = F.samples_num_count(self.client_dict, self.num_clients)

    def _perform_partition(self):
        if self.partition == "noniid-#label":
            # label-distribution-skew:quantity-based
            client_dict = F.label_skew_quantity_based_partition(self.targets, self.num_clients,
                                                                self.num_classes,
                                                                self.major_classes_num)

        elif self.partition == "noniid-labeldir":
            # label-distribution-skew:distributed-based (Dirichlet)
            client_dict = F.hetero_dir_partition(self.targets, self.num_clients, self.num_classes,
                                                 self.dir_alpha,
                                                 min_require_size=10)

        elif self.partition == "unbalance":
            # quantity-skew (Dirichlet)
            client_sample_nums = F.dirichlet_unbalance_split(self.num_clients, self.num_samples,
                                                             self.dir_alpha)
            client_dict = F.homo_partition(client_sample_nums, self.num_samples)

        else:
            # IID
            client_sample_nums = F.balance_split(self.num_clients, self.num_samples)
            client_dict = F.homo_partition(client_sample_nums, self.num_samples)

        return client_dict

    def __getitem__(self, index):
        return self.client_dict[index]

    def __len__(self):
        return len(self.client_dict)