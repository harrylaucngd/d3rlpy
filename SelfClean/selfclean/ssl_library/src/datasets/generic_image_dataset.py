import os
from pathlib import Path
from typing import Sequence, Union

import pandas as pd
import torch
from loguru import logger
from PIL import Image

from ...src.datasets.base_dataset import BaseDataset


class GenericImageDataset(BaseDataset):
    """Generic image dataset."""

    IMG_COL = "img_path"
    LBL_COL = "diagnosis"

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "data/dataset/",
        transform=None,
        val_transform=None,
        return_path: bool = False,
        image_extensions: Sequence = ("*.png", "*.jpg", "*.JPEG"),
        **kwargs,
    ):
        """
        Initializes the dataset.

        Sets the correct path for the needed arguments.

        Parameters
        ----------
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
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise ValueError(f"Image path must exist, path: {self.dataset_dir}")

        # create dicts for retreiving imgs and masks
        l_files = []
        for extension in image_extensions:
            l_files.extend(
                GenericImageDataset.find_files_with_extension(
                    directory_path=dataset_dir,
                    extension=extension,
                )
            )

        # create the metadata dataframe
        if len(set(l_files)) != len(l_files):
            logger.info(f"Caution! There are duplicate files.")
        self.meta_data = pd.DataFrame(set(l_files))
        self.meta_data.columns = [self.IMG_COL]
        self.meta_data["img_name"] = self.meta_data[self.IMG_COL].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0]
        )
        self.meta_data[self.LBL_COL] = self.meta_data[self.IMG_COL].apply(
            lambda x: Path(x).parents[0].name
        )
        self.meta_data.reset_index(drop=True, inplace=True)
        int_lbl, lbl_mapping = pd.factorize(self.meta_data[self.LBL_COL])
        self.LBL_COL = f"lbl_{self.LBL_COL}"
        self.meta_data[self.LBL_COL] = int_lbl

        # global configs
        self.return_path = return_path
        self.classes = list(lbl_mapping)
        self.n_classes = len(self.classes)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = self.meta_data.loc[self.meta_data.index[index], self.IMG_COL]
        image = Image.open(img_name)
        image = image.convert("RGB")
        if self.transform and self.training:
            image = self.transform(image)
        elif self.val_transform and not self.training:
            image = self.val_transform(image)

        diagnosis = self.meta_data.loc[self.meta_data.index[index], self.LBL_COL]
        if self.return_path:
            return image, img_name, int(diagnosis)
        else:
            return image, int(diagnosis)
