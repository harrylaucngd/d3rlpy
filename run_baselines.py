import d3rlpy
from d3rlpy.datasets import get_d4rl
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

# Define the environments and algorithms
# environments = ['hopper', 'halfcheetah', 'walker2d']
environments = ['halfcheetah']
# algorithms = ['cql', 'bcq', 'iql']
algorithms = ['bcq']

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

    # params = {'bandwidth': np.logspace(-1, 0, 2)}
    # print(f"params: {params}")
    # grid = GridSearchCV(KernelDensity(), params)
    # grid.fit(all_states)
    #
    # # Best bandwidth from GridSearchCV
    # best_bandwidth = grid.best_estimator_.bandwidth
    # print(f"best_bandwidth: {best_bandwidth}")
    # exit()

    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(all_states)

    # Step 4: Calculate the density for a new state
    # state = dataset.episodes[0].observations[0, :]  # Replace with your new state to estimate density
    state = np.random.randn(17)
    log_density = kde.score_samples(state.reshape(1, -1))
    density = np.exp(log_density)

    # Step 5: Compute pseudo-counts
    pseudo_count = 1.0 / density

    # Step 6: Calculate exploration bonus
    exploration_bonus = 1.0 / np.sqrt(pseudo_count)

    print(f"Density: {density}")
    print(f"Pseudo-Count: {pseudo_count}")
    print(f"Exploration Bonus: {exploration_bonus}")
    exit()


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
              evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)})

    # Evaluate the model (you can customize the evaluation here)
    # For instance, you might want to use a separate evaluation environment or dataset.

if __name__ == '__main__':
    for env in environments:
        for algo in algorithms:
            print(f"Starting experiment for {env} with {algo}")
            run_experiment(env, algo)
