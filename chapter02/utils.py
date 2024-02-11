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


def plot_experiment(agents, data):
    plt.figure(layout="constrained")

    plt.subplot(211)
    for key in agents.keys():
        plt.plot(key, data=data["avg_reward"])
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.grid(True)

    plt.subplot(212)
    for key in agents.keys():
        plt.plot(key, data=data["optimal_action"])
    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")
    plt.legend()
    plt.grid(True)

    plt.show()
