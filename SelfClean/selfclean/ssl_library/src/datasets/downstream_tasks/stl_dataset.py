from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision


class STL10Dataset(torchvision.datasets.STL10):
    """STL-10 dataset."""

    LBL_COL = "label"

    def __init__(
        self,
        root: str,
        split: str = "train",
        folds: Optional[int] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        high_quality: bool = False,
        **kwargs,
    ):
        super().__init__(
            root=root,
            split=split,
            folds=folds,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.meta_data = pd.DataFrame(
            np.arange(self.data.shape[0]),
            columns=["data index"],
        )
        self.meta_data["label"] = self.labels
        if high_quality:
            irr_indices = [1473, 56, 298, 1741, 1021, 4247]

            gray_indices = [70, 215, 422, 488, 602, 608, 844, 1585, 1624, 1891, 2026]
            gray_indices += [2490, 2684, 2693, 2747, 2873, 2949, 3177, 3212, 3246]
            gray_indices += [3343, 3587, 4034, 4041, 4065, 4156, 4416, 4472, 4753]
            gray_indices += [4893, 4941, 4993]

            dup_indices = [3310, 3309, 1092, 1170, 3865, 2820, 3845, 2429, 3376]
            dup_indices += [2987, 3443, 2733, 4699, 4301, 4166, 1894, 1975, 3706]
            dup_indices += [2563, 2700, 2733, 4323, 4850, 4603, 4512, 3021, 4327]
            dup_indices += [2409, 3700, 1632, 3506, 951, 483, 1940, 4898, 4699]
            dup_indices += [3181, 2980, 792, 1074, 3689, 4869, 2131, 3473, 2492]
            dup_indices += [4327, 4739, 1629, 4767, 2825, 4327, 2759, 4976, 1516]
            dup_indices += [4609, 4348, 2765, 3674, 1951, 3240, 4898, 2542, 4869]
            dup_indices += [4187, 732, 3809, 4435, 2980, 4321, 4898, 4692, 2157]
            dup_indices += [3021, 4265, 4898, 3629, 4805]
            dup_indices += [2745, 3068, 3575, 3037, 1003]

            dqa_issues = list(set(dup_indices + gray_indices + irr_indices))
            self.data = np.delete(self.data, dqa_issues, axis=0)
            self.labels = np.delete(self.labels, dqa_issues, axis=0)
            self.meta_data = self.meta_data.drop(dqa_issues)
            self.meta_data.reset_index(drop=True, inplace=True)
        # global configs
        self.n_classes = len(self.classes)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        rets = super().__getitem__(index=index)
        rets = (rets[0], "", rets[1])
        return rets

    @staticmethod
    def collate_fn(batch):
        return torch.utils.data.dataloader.default_collate(batch)
