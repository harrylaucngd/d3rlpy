import h5py
import numpy as np

def remove_duplicates(actions, observations, rewards, terminals, timeouts):
    unique_trajectories = []
    seen_trajectories = set()
    
    start_idx = 0
    for i, (terminal, timeout) in enumerate(zip(terminals, timeouts)):
        if terminal or timeout:
            end_idx = i + 1
            trajectory = (
                tuple(map(tuple, actions[start_idx:end_idx])),
                tuple(map(tuple, observations[start_idx:end_idx])),
                tuple(rewards[start_idx:end_idx]),
                tuple(terminals[start_idx:end_idx]),
                tuple(timeouts[start_idx:end_idx])
            )
            if trajectory not in seen_trajectories:
                seen_trajectories.add(trajectory)
                unique_trajectories.append((actions[start_idx:end_idx],
                                            observations[start_idx:end_idx],
                                            rewards[start_idx:end_idx],
                                            terminals[start_idx:end_idx],
                                            timeouts[start_idx:end_idx]))
            start_idx = end_idx
            
    return unique_trajectories

def remove_low_return_trajectories(hdf5_path, output_path, keep_ratio=0.5):
    # 打开HDF5文件
    with h5py.File(hdf5_path, 'r') as f:
        actions = f['actions'][:]
        observations = f['observations'][:]
        rewards = f['rewards'][:]
        terminals = f['terminals'][:]
        timeouts = f['timeouts'][:]

    # 计算每条轨迹的总回报
    trajectory_returns = []
    start_idx = 0
    for i, (terminal, timeout) in enumerate(zip(terminals, timeouts)):
        if terminal or timeout:
            end_idx = i + 1
            trajectory_returns.append(np.sum(rewards[start_idx:end_idx]))
            start_idx = end_idx

    # 按返回值排序并确定要保留的轨迹数量
    sorted_indices = np.argsort(trajectory_returns)
    num_trajectories_to_keep = int(len(trajectory_returns) * keep_ratio)
    keep_indices = sorted_indices[-num_trajectories_to_keep:]

    # 筛选高返回率的轨迹
    new_actions, new_observations, new_rewards, new_terminals, new_timeouts = [], [], [], [], []
    start_idx = 0
    trajectory_idx = 0
    for i, (terminal, timeout) in enumerate(zip(terminals, timeouts)):
        if terminal or timeout:
            end_idx = i + 1
            if trajectory_idx in keep_indices:
                new_actions.extend(actions[start_idx:end_idx])
                new_observations.extend(observations[start_idx:end_idx])
                new_rewards.extend(rewards[start_idx:end_idx])
                new_terminals.extend(terminals[start_idx:end_idx])
                new_timeouts.extend(timeouts[start_idx:end_idx])
            start_idx = end_idx
            trajectory_idx += 1

    # 去重
    unique_trajectories = remove_duplicates(new_actions, new_observations, new_rewards, new_terminals, new_timeouts)
    
    # 将去重后的数据写入新的HDF5文件
    with h5py.File(output_path, 'w') as f:
        actions, observations, rewards, terminals, timeouts = [], [], [], [], []
        for trajectory in unique_trajectories:
            actions.extend(trajectory[0])
            observations.extend(trajectory[1])
            rewards.extend(trajectory[2])
            terminals.extend(trajectory[3])
            timeouts.extend(trajectory[4])
        f.create_dataset('actions', data=actions)
        f.create_dataset('observations', data=observations)
        f.create_dataset('rewards', data=rewards)
        f.create_dataset('terminals', data=terminals)
        f.create_dataset('timeouts', data=timeouts)

# 3个数据集各生成10种比例的trimmed datasets
datasets = ["halfcheetah", "hopper", "walker"]
ratios = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
for dataset in datasets:
    for keep_ratio in ratios:
        hdf5_path = f'datasets/{dataset}_mixed.hdf5'
        output_path = f'trimmed_datasets/{dataset}_mixed_{keep_ratio}.hdf5'
        remove_low_return_trajectories(hdf5_path, output_path, keep_ratio)
