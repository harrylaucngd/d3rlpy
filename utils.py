import numpy as np
import h5py
from d3rlpy.types import NDArray
from d3rlpy.datasets import MDPDataset


dataset_name = {
    "halfcheetah-medium-replay-v0": "halfcheetah_mixed",
    "hopper-medium-replay-v0": "hopper_mixed",
    "walker2d-medium-replay-v0": "walker_mixed",
    }


def hdf5_to_dict(hdf5_group):
    result = {}
    for key, item in hdf5_group.items():
        if isinstance(item, h5py.Group):
            result[key] = hdf5_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            result[key] = np.array(item[()].tolist())
        else:
            result[key] = item[()]
    return result


def get_dataset(name):
    try:
        with h5py.File(name, 'r') as hdf:
            dataset_dict = hdf5_to_dict(hdf)
    except Exception as e:
        print(f"Error reading {name}: {e}")
    
    return dataset_dict


def get_trimmed_dataset(dataset, keep_ratio):
    name = f"trimmed_datasets/{dataset_name[dataset]}_{keep_ratio}.hdf5"
    raw_dataset: dict[str, np.ndarray] = get_dataset(name)  # type: ignore

    observations = raw_dataset["observations"]
    actions = raw_dataset["actions"]
    rewards = raw_dataset["rewards"]
    terminals = raw_dataset["terminals"]
    timeouts = raw_dataset["timeouts"]

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )
    return dataset