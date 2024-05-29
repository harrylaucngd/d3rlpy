import d3rlpy
from d3rlpy.datasets import get_d4rl

# Define the environments and algorithms
environments = ['hopper', 'halfcheetah', 'walker2d']
algorithms = ['cql', 'bcq', 'iql']

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
