from tqdm import tqdm
import d3rlpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from model import lstm_seq2seq
from utils import get_trimmed_dataset
import h5py
import argparse


parser = argparse.ArgumentParser(description='LSTM Generating New Data')
parser.add_argument("--model_name", type=str, default="hopper_ae", help="Name of the model")
parser.add_argument("--dataset", type=str, default="hopper-medium-replay-v0", help="Name of the dataset")
parser.add_argument("--dataset_size", type=int, default=-1, help="Size of the dataset,if -1 use all data")
parser.add_argument("--model_type", type=str, default="autoencoder", help="Type of the model")
parser.add_argument("--seq_len", type=int, default=10, help="Load the model from the saved model")
args = parser.parse_args()

KEEP_RATIO = 0.1

input_dim_table = {
    "hopper-medium-replay-v0": 15,
    "halfcheetah-medium-replay-v0": 24,
    "walker2d-medium-replay-v0": 24,
    }

trimmed_dataset_table = {
    "hopper-medium-replay-v0": "hopper",
    "halfcheetah-medium-replay-v0": "halfcheetah",
    "walker2d-medium-replay-v0": "walker",
}

input_dim = input_dim_table[args.dataset]  # 每个时间步的向量维度
hidden_dim = 128  # 隐藏层维度
latent_dim = 64  # 自编码器学习的低维表示
batch_size = 256  # 批处理大小


class VariableLengthDataset(Dataset):
    def __init__(self, trajectories):
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def prepare_data():
    dataset, _ = d3rlpy.datasets.get_dataset(args.dataset)
    dataset = get_trimmed_dataset(args.dataset, KEEP_RATIO)
    if args.dataset_size != -1:
        traj_dataset = dataset._buffer._episodes[:args.dataset_size]
    else:
        traj_dataset = dataset._buffer._episodes
    traj_num = len(traj_dataset)

    all_traj_repsentations = []
    for i in tqdm(range(traj_num)):
        for j in range(0,len(traj_dataset[i].observations),args.seq_len):
            seq_len = len(traj_dataset[i].observations)-j if len(traj_dataset[i].observations)-j<args.seq_len else args.seq_len
            obs= torch.tensor(traj_dataset[i].observations[j:j+seq_len])
            action = torch.tensor(traj_dataset[i].actions[j:j+seq_len])
            reward = torch.tensor(traj_dataset[i].rewards[j:j+seq_len])
            
            traj_representation = torch.cat((obs, action,reward), dim=-1)
            all_traj_repsentations.append(traj_representation)
    dataset = VariableLengthDataset(all_traj_repsentations)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: nn.utils.rnn.pad_sequence(x, batch_first=True))
    return data_loader


data_loader = prepare_data()


def read_hdf5_to_dict(file_path):
    data_dict = {}
    
    def recursively_load_dict_contents(group, dict_obj):
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                dict_obj[key] = item[()]
            elif isinstance(item, h5py.Group):
                dict_obj[key] = {}
                recursively_load_dict_contents(item, dict_obj[key])
    
    with h5py.File(file_path, 'r') as hdf_file:
        recursively_load_dict_contents(hdf_file, data_dict)
    
    return data_dict


def generate(input_dim, hidden_dim):
    model = lstm_seq2seq(input_dim, hidden_dim)
    model.eval()
    model = torch.load(f"./temp_result/{args.model_name}_best.pt").to("cuda")
    reconstructed_trajectories = []
    with torch.no_grad():
        for batch in data_loader:
            batch_trajectories = batch.permute(1,0,2).to("cuda").float()
            reconstructed = model(input_batch=batch_trajectories,target_batch=batch_trajectories, training_prediction='teacher_forcing', teacher_forcing_ratio=0.5).to("cuda").permute(1,0,2)
        reconstructed_trajectories.extend(reconstructed.cpu().numpy())

    dataset = read_hdf5_to_dict(f"./trimmed_datasets/{trimmed_dataset_table[args.dataset]}_mixed_{KEEP_RATIO}.hdf5")
    obs_shape = dataset["observations"].shape[-1]
    action_shape = dataset["actions"].shape[-1]
    reward_shape = dataset["rewards"].shape[-1]
    reconstructed_dataset = {
        "observations": [],
        "actions": [],
        "rewards": [],
        }
    for traj in reconstructed_trajectories:
        for frame in traj:
            reconstructed_dataset["observations"].append(frame[:obs_shape])
            reconstructed_dataset["actions"].append(frame[obs_shape:(obs_shape+action_shape)])
            reconstructed_dataset["rewards"].append(frame[-1])
    dataset["observations"] = reconstructed_dataset["observations"]
    dataset["actions"] = reconstructed_dataset["actions"]
    dataset["rewards"] = reconstructed_dataset["rewards"]
    with h5py.File(f"./generated_datasets/{args.dataset}_reconstructed_{KEEP_RATIO}.hdf5", 'w') as h5file:
        for key, value in dataset.items():
            h5file.create_dataset(key, data=value)


if __name__ == '__main__':
    generate(input_dim, hidden_dim)