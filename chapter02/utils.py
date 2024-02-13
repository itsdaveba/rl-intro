import matplotlib.pyplot as plt
from bandit import run_episode


def run_experiment(env, agents, seed=None):
    data = {"avg_reward": {}, "optimal_action": {}}
    seeds = [seed + i if seed is not None else None for i in range(len(agents))]
    for i, (key, agent) in enumerate(agents.items()):
        rewards, optimals = run_episode(env, agent, seeds[i])
        data["avg_reward"][key] = rewards.mean(axis=1)
        data["optimal_action"][key] = 100 * optimals.mean(axis=1)
    return data


def plot_experiment(data):
    plt.figure(layout="constrained")

    plt.subplot(211)
    for key in data["avg_reward"].keys():
        plt.plot(key, data=data["avg_reward"])
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.grid(True)

    plt.subplot(212)
    for key in data["optimal_action"].keys():
        plt.plot(key, data=data["optimal_action"])
    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")
    plt.legend()
    plt.grid(True)

    plt.show()


def run_parameter_study(env, study, last_steps=None, seed=None):
    data = {"avg_reward": {}}
    if last_steps is None:
        last_steps = env.max_episode_steps
    data["last_steps"] = last_steps
    seeds = [seed + i if seed is not None else None for i in range(len(study))]
    for i, (alg_key, alg) in enumerate(study.items()):
        key = list(alg["parameter"].keys())[0]
        params = list(alg["parameter"].values())[0]
        kwargs = alg.get("constant_parameters", {})
        avg_rewards = []
        for param in params:
            kwargs.update([(key, param)])
            agent = alg["entry_point"](env.k, env.num_envs, **kwargs)
            rewards, _ = run_episode(env, agent, seeds[i])
            avg_rewards.append(rewards[-last_steps:].mean())
        data["avg_reward"][alg_key] = {"params": params, alg_key: avg_rewards}
    return data


def plot_parameter_study(data):
    for key, val in data["avg_reward"].items():
        plt.plot("params", key, data=val)
    plt.xscale("log", base=2)
    plt.xlabel("Parameter")
    plt.ylabel(f"Average reward over last {data['last_steps']} steps")
    plt.legend()
    plt.grid(True)
    plt.show()
