from d3rlpy.logging import WanDBAdapterFactory
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys
import d3rlpy
sys.path.append("/home/liwentao/d3rlpy/SelfClean")
from selfclean import SelfClean
from selfclean.cleaner.selfclean import PretrainingType
from selfclean.utils.data_downloading import get_imagenette
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Data Cleaning For ORL')
parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gpu", type=int, default=2)
parser.add_argument("--use_cleaning", action="store_true", help="Use Weights & Biases for logging")
parser.add_argument("--dataset", type=str, default="hopper-medium-replay-v0", help="Name of the dataset")
parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset,if -1 use all data")
parser.add_argument("--trim_ratio", type=float, default=0.1, help="Trim ratio for data cleaning")
args = parser.parse_args()

class VariableLengthDataset(Dataset):
    def __init__(self, trajectories):
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]

def prepare_data():
    ori_dataset, env = d3rlpy.datasets.get_dataset("hopper-medium-replay-v0")
    if args.dataset_size != -1:
        ori_dataset._buffer._transitions = ori_dataset._buffer[:args.dataset_size]
        ori_dataset._buffer._transition_count = args.dataset_size

    traj_num = len(ori_dataset._buffer)
    action_size = ori_dataset._dataset_info.action_signature.shape[0][0]
    obs_size = ori_dataset._dataset_info.observation_signature.shape[0][0]

    print(f"Number of trajectories: {traj_num}")
    all_traj_repsentations = []
    traj_labels = []
    for i in tqdm(range(traj_num)):
        obs= torch.tensor(ori_dataset._buffer[i][0].observations)
        action = torch.tensor(ori_dataset._buffer[i][0].actions)
        reward = torch.tensor(ori_dataset._buffer[i][0].rewards)
        # import pdb; pdb.set_trace()
        all_traj_repsentations.append(torch.cat((obs, action,reward), dim=-1))
    dataset = VariableLengthDataset(all_traj_repsentations)

    # 数据集划分

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=lambda x: nn.utils.rnn.pad_sequence(x, batch_first=True))
    
    return ori_dataset,dataset,dataloader,env

ori_dataset,dataset,dataloader,env = prepare_data()

if args.use_cleaning:
    selfclean = SelfClean(
        plot_top_N=7,
        auto_cleaning=True,
    )

    issues = selfclean.run_on_dataset(
        dataset=dataset,
        pretraining_type=PretrainingType.AE,
        epochs=10,
        batch_size=16,
        save_every_n_epochs=1,
        # dataset_name=dataset_name,
        work_dir=None,
        dataloader=dataloader,
    )
    # reset to our visualisation augmentation
    df_near_duplicates = issues.get_issues("near_duplicates", return_as_df=True)
    print(df_near_duplicates.head())

    df_irrelevants = issues.get_issues("irrelevants", return_as_df=True)
    print(df_irrelevants.head())

    # 剪去得分较低的样本，其中near_duplicates是n*n长度，irrelevants是n长度
    irrelevant_indices = df_irrelevants["indices"].values.tolist()
    irrelevant_trim_ids = irrelevant_indices[:int(len(irrelevant_indices) * args.trim_ratio)]
    # TODO 怎么删除近邻重复的样本
    near_duplicates_indices = df_near_duplicates["indices_1"].values.tolist()
    near_duplicates_trim_ids = near_duplicates_indices[:int(len(near_duplicates_indices) * args.trim_ratio**3)]
    near_duplicates_trim_ids = list(set(near_duplicates_trim_ids))

    combined_list = irrelevant_trim_ids + near_duplicates_trim_ids
    trim_ids_set = set(combined_list)

    all_ids = set(range(args.dataset_size))

    # Get the difference between the two sets
    remaining_ids = all_ids - trim_ids_set

    # Convert the result back to a list, if needed
    remaining_ids_list = list(remaining_ids)

ori_dataset._buffer._transitions = [ori_dataset._buffer._transitions[i] for i in remaining_ids_list]
ori_dataset._buffer._transition_count = args.dataset_size

# Offline RL Training
d3rlpy.seed(args.seed)
d3rlpy.envs.seed_env(env, args.seed)

encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])

if "medium-v0" in args.dataset:
    conservative_weight = 10.0
else:
    conservative_weight = 5.0

cql = d3rlpy.algos.CQLConfig(
    actor_learning_rate=1e-4,
    critic_learning_rate=3e-4,
    temp_learning_rate=1e-4,
    actor_encoder_factory=encoder,
    critic_encoder_factory=encoder,
    batch_size=256,
    n_action_samples=10,
    alpha_learning_rate=0.0,
    conservative_weight=conservative_weight,
).create(device=args.gpu)

if args.use_cleaning:
    logger_adapter = WanDBAdapterFactory("Data cleaning for CQL on Hopper-medium-replay-v0")
else:
    logger_adapter = WanDBAdapterFactory("Pure CQL on Hopper-medium-replay-v0")

cql.fit(
    ori_dataset,
    n_steps=50000,
    n_steps_per_epoch=1000,
    save_interval=10,
    logger_adapter = logger_adapter,
    evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
    experiment_name=f"CQL_{args.dataset}_{args.seed}",
)