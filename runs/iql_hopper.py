import argparse
from utils import get_trimmed_dataset

from torch.optim.lr_scheduler import CosineAnnealingLR

import d3rlpy


def main(keep_ratio = 0.5) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hopper-medium-replay-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)
    dataset = get_trimmed_dataset(args.dataset, keep_ratio)

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    reward_scaler = d3rlpy.preprocessing.ReturnBasedRewardScaler(
        multiplier=1000.0
    )

    iql = d3rlpy.algos.IQLConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        batch_size=256,
        weight_temp=3.0,
        max_weight=100.0,
        expectile=0.7,
        reward_scaler=reward_scaler,
    ).create(device=args.gpu)

    # workaround for learning scheduler
    iql.build_with_dataset(dataset)
    assert iql.impl
    scheduler = CosineAnnealingLR(
        iql.impl._modules.actor_optim,  # pylint: disable=protected-access
        500000,
    )

    def callback(algo: d3rlpy.algos.IQL, epoch: int, total_step: int) -> None:
        scheduler.step()

    iql.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=10,
        callback=callback,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"IQL_{args.dataset}_{keep_ratio}_{args.seed}",
    )


if __name__ == "__main__":
    for keep_ratio in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]:
        main(keep_ratio)
