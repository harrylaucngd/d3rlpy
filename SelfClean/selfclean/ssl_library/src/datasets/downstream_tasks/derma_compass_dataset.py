import os
from enum import Enum
from glob import glob
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ....src.datasets.base_dataset import BaseDataset


class DermaCompassLabel(Enum):
    # col name of label code, col name of label
    PRIMARY = "lbl_primary", "primary"
    SECONDARY = "lbl_secondary", "secondary"


class DermaCompassDataset(BaseDataset):
    """DermaCompass dataset."""

    IMG_COL = "title"
    LBL_COL = "label"

    def __init__(
        self,
        csv_file: Union[str, Path] = "data/DermaCompass/data.csv",
        dataset_dir: Union[str, Path] = "data/DermaCompass/",
        transform=None,
        val_transform=None,
        label_col: DermaCompassLabel = DermaCompassLabel.PRIMARY,
        return_path: bool = False,
        **kwargs,
    ):
        """
        Initializes the dataset.

        Sets the correct path for the needed arguments.

        Parameters
        ----------
        csv_file : str
            Path to the csv file with metadata, including annotations.
        dataset_dir : str
            Directory with all the images.
        transform : Union[callable, optional]
            Optional transform to be applied to the images.
        val_transform : Union[callable, optional]
            Optional transform to be applied to the images when in validation mode.
        return_path : bool
            If the path of the image should be returned or not.
        """
        super().__init__(transform=transform, val_transform=val_transform, **kwargs)
        # check if the dataset path exists
        self.csv_file = Path(csv_file)
        if not self.csv_file.exists():
            raise ValueError(f"CSV metadata path must exist, path: {self.csv_file}")
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise ValueError(f"Image path must exist, path: {self.dataset_dir}")
        # transform the dataframe for better loading
        imageid_path_dict = {
            os.path.splitext(os.path.basename(x))[0]: x
            for x in glob(os.path.join(self.dataset_dir, "*", "*.jpg"))
        }
        # load the metadata
        self.meta_data = pd.DataFrame(pd.read_csv(csv_file, index_col=0))
        self.meta_data["path"] = self.meta_data[self.IMG_COL].map(imageid_path_dict.get)
        self.meta_data = self.meta_data.dropna(subset=["path"])
        self.meta_data = self.meta_data.drop_duplicates(subset="title")
        self.meta_data[DermaCompassLabel.PRIMARY.value[1]] = self.meta_data[
            DermaCompassLabel.PRIMARY.value[1]
        ].replace(np.nan, "undefined")
        self.meta_data[DermaCompassLabel.SECONDARY.value[1]] = self.meta_data[
            DermaCompassLabel.SECONDARY.value[1]
        ].replace(np.nan, "undefined")
        # transform the string labels into categorical values
        self.meta_data[DermaCompassLabel.PRIMARY.value[0]] = pd.factorize(
            self.meta_data[DermaCompassLabel.PRIMARY.value[1]]
        )[0]
        self.meta_data[DermaCompassLabel.SECONDARY.value[0]] = pd.factorize(
            self.meta_data[DermaCompassLabel.SECONDARY.value[1]]
        )[0]
        # global configs
        self.return_path = return_path
        self.LBL_COL = label_col.value[0]
        self.classes = self.meta_data[label_col.value[1]].unique().tolist()
        self.n_classes = len(self.classes)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.meta_data.loc[self.meta_data.index[idx], "path"]
        image = Image.open(img_name)
        image = image.convert("RGB")
        if self.transform and self.training:
            image = self.transform(image)
        elif self.val_transform and not self.training:
            image = self.val_transform(image)

        diagnosis = self.meta_data.loc[self.meta_data.index[idx], self.LBL_COL]
        if self.return_path:
            return image, img_name, int(diagnosis)
        else:
            return image, int(diagnosis)
