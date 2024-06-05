import torch
from torchvision import datasets, transforms
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys
sys.path.append("../")
from selfclean import SelfClean
from selfclean.cleaner.selfclean import PretrainingType
from selfclean.utils.data_downloading import get_imagenette


pre_computed_path = Path("../assets/pre_trained_models")

dataset_name = "ImageNette"
data_path = Path("../data/")
dataset, df = get_imagenette(
   root_path=data_path, return_dataframe=True, transform=transforms.Resize((256, 256))
)

selfclean = SelfClean(
    plot_top_N=7,
    auto_cleaning=True,
)
issues = selfclean.run_on_dataset(
    dataset=copy.copy(dataset),
    pretraining_type=PretrainingType.DINO,
    epochs=10,
    batch_size=16,
    save_every_n_epochs=1,
    dataset_name=dataset_name,
    work_dir=pre_computed_path,
)
# reset to our visualisation augmentation
dataset.transforms = None

df_near_duplicates = issues.get_issues("near_duplicates", return_as_df=True)
df_near_duplicates.head()