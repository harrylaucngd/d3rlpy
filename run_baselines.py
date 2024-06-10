import argparse
import d3rlpy
from d3rlpy.datasets import get_d4rl
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Define the environments and algorithms
# environments = ['hopper', 'halfcheetah', 'walker2d']
environments = ['halfcheetah']
# algorithms = ['cql', 'bcq', 'iql']
algorithms = ['bcq']

def run_experiment(env_name, algo_name, args):
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

    # print(len(dataset.episodes))
    # print(len(dataset.episodes[0]))
    # print(dataset.episodes[0][0])
    # for episode in dataset.episodes:
    #     for i in range(episode.size()):
    #         episode.observation_signature
    list_of_states = []
    for episode in dataset.episodes:
        list_of_states.append(episode.observations)
    all_states = np.vstack(list_of_states)
    print(f"all_states: {all_states.shape}")
    scaler = StandardScaler()
    states_normalized = scaler.fit_transform(all_states)

    bandwidth = 0.4  # best bandwidth, making pseudo count ~ 4, maximizing the rate of std_count / mean_count
    alpha = args.alpha
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(states_normalized)
    print(f"total episode: {len(dataset.episodes)}")
    for i, episode in enumerate(dataset.episodes):
        print(f"i: {i}")
        print(f"np.mean(episode.rewards): {np.mean(episode.rewards)}")
        mean_abs_reward = np.mean(np.abs(episode.rewards))
        # for i in range(episode.size() - 1):
        next_observations = episode.observations[1:, :]
        # print(f"next_observations: {next_observations.shape}")
        states_normalized = scaler.transform(next_observations)
        # print(f"states_normalized: {states_normalized.shape}")
        log_density = kde.score_samples(states_normalized)
        # print(f"log_density: {log_density.shape}")
        density = np.exp(log_density)
        pseudo_count = density * all_states.shape[0]
        exploration_bonus = 1.0 / np.sqrt(pseudo_count)
        # print(f"exploration_bonus: {exploration_bonus.shape}")
        # print(f"episode.rewards: {episode.rewards.shape}")
        episode.rewards[:-1, :] += np.reshape(alpha * (exploration_bonus / 0.5 * mean_abs_reward), (-1, 1))
        print(f"np.mean(episode.rewards): {np.mean(episode.rewards)}")

    # params = {'bandwidth': np.logspace(-1, 0, 2)}
    # print(f"params: {params}")
    # grid = GridSearchCV(KernelDensity(), params)
    # grid.fit(all_states)
    #
    # # Best bandwidth from GridSearchCV
    # best_bandwidth = grid.best_estimator_.bandwidth
    # print(f"best_bandwidth: {best_bandwidth}")
    # exit()

    # exit()
    # print(dataset.episodes[0].rewards[0])
    # dataset.episodes[0].rewards[0] = 0
    # print(dataset.episodes[0].rewards[0])
    # # print(dataset.transition_picker(dataset.episodes[0], 0))
    # exit()

    # Fit the model
    model.fit(dataset,
              n_steps=1000000,
              n_steps_per_epoch=10000,
              evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
              experiment_name=f"{env_name}-{algo_name}-{args.run_name}")

    # Evaluate the model (you can customize the evaluation here)
    # For instance, you might want to use a separate evaluation environment or dataset.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run D3RLPy experiment')
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()

    for env in environments:
        for algo in algorithms:
            print(f"Starting experiment for {env} with {algo}")
            run_experiment(env, algo, args)
