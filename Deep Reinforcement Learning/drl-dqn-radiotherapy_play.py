import matplotlib.pyplot as plt
import numpy as np
import torch
from lib import environ, dqlmodel


class Agent:
    def __init__(self, env):
        self.env = env
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None
        rad_day = Action.radio_day(self.env.daily_week)
        if rad_day:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_a = self.state
                state_v = torch.from_numpy(state_a).float().unsqueeze(0).to(device)
                #state_v = torch.tensor(state_a).to(device)
                q_vals_v = net(state_v)
                _, act_v = torch.max(q_vals_v, dim=1)
                action = act_v
        else:
            action = 0

        # do step in the environment
        new_state, reward, is_done = self.env.step(action)
        self.total_reward += reward

        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            # self._reset()
        return done_reward


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = environ.Env(dailydose_max=2, tot_dose_max=30, num_fractions=30,
                      numba=True, state_space="Continuous")

    net = dqlmodel.SimpleFFDQN(env.observation.statespace.n,
                               env.action_space.n).to(device)

    state = torch.load("-best_-10.dat", map_location=torch.device(device))
    net.load_state_dict(state)
    agent = Agent(env)

    epochs_idx = 0
    for i in range(40):
        epochs_idx += 1

        reward = agent.play_step(net, device=device)
    plt.figure()
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['figure.figsize'] = [10, 5.5]
    giorni = epochs_idx
    t = np.linspace(0, giorni, giorni * 1440 + 1)
    plt.plot(t, agent.env.observation.tumor_evolution, "b", label='Tumor Evolution')
    plt.plot(t, agent.env.observation.healthy_evolution, "r", label='Health Cell Evolution')
    plt.xlabel("Time (day)")
    plt.title("Radiation Therapy Evolution")
    plt.legend()
