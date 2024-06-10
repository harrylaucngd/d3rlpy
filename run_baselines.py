import d3rlpy
from d3rlpy.datasets import get_d4rl
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

# Define the environments and algorithms
environments = ['hopper', 'halfcheetah', 'walker2d']
algorithms = ['cql', 'bcq', 'iql']
bandwidth_dict = {'hopper': 0.6, 'halfcheetah': 0.4, 'walker2d': 0.4}

def run_experiment(env_name, algo_name):
    # Load the dataset
    dataset_name = f'{env_name}-medium-replay-v0'
    dataset, env = get_d4rl(dataset_name)

    # Create an algorithm object based on the input
    if algo_name == 'bcq':
        model = d3rlpy.algos.BCQConfig().create(device="cuda:0")
    elif algo_name == 'cql':
        model = d3rlpy.algos.CQLConfig().create(device="cuda:0")
    elif algo_name == 'iql':
        model = d3rlpy.algos.IQLConfig().create(device="cuda:0")
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")

    list_of_states = []
    for episode in dataset.episodes:
        list_of_states.append(episode.observations)
    all_states = np.vstack(list_of_states)
    print(f"all_states: {all_states.shape}")
    scaler = StandardScaler()
    states_normalized = scaler.fit_transform(all_states)

    bandwidth = bandwidth_dict[env_name]  # best bandwidth, making pseudo count ~ 4, maximizing the rate of std_count / mean_count
    epsilon = 0.1
    alpha = 0.5
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(states_normalized)
    print(f"total episode: {len(dataset.episodes)}")
    for i, episode in enumerate(dataset.episodes):
        print(f"i: {i}")
        print(f"np.mean(episode.rewards): {np.mean(episode.rewards)}")
        mean_abs_reward = np.mean(np.abs(episode.rewards))
        next_observations = episode.observations[1:, :]
        states_normalized = scaler.transform(next_observations)
        log_density = kde.score_samples(states_normalized)
        density = np.exp(log_density)
        pseudo_count = density * all_states.shape[0]
        exploration_bonus = 1.0 / (np.sqrt(pseudo_count) + epsilon)
        episode.rewards[:-1, :] += np.reshape(alpha * (exploration_bonus / 0.5 * mean_abs_reward), (-1, 1))
        print(f"np.mean(episode.rewards): {np.mean(episode.rewards)}")

    # Fit the model
    model.fit(dataset,
              n_steps=1000000,
              n_steps_per_epoch=10000,
              evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)})

    # Evaluate the model (you can customize the evaluation here)
    # For instance, you might want to use a separate evaluation environment or dataset.

if __name__ == '__main__':

    for env in environments:
        for algo in algorithms:
            print(f"Starting experiment for {env} with {algo}")
            run_experiment(env, algo)
