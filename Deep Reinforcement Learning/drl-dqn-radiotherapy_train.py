import collections
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from lib import dqlmodel, environ
from tensorboardX import SummaryWriter

BATCH_SIZE = 64
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 2000
SYNC_TARGET_FRAMES = 3000

GAMMA = 0.999
LEARNING_RATE = 1e-4

EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 300000

NUM_EPISODES = 500000

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'new_state', "done"])


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        states, actions, rewards, next_states, dones = \
            zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions, dtype=np.int32), \
            np.array(rewards, dtype=np.float32), \
            np.array(next_states),  np.array(dones, dtype=np.uint8)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        rad_day = environ.Action.radio_day(self.env.daily_week)
        if rad_day:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_a = self.state
                state_v = torch.from_numpy(
                    state_a).float().unsqueeze(0).to(device)
                #state_v = torch.tensor(state_a).to(device)
                q_vals_v = net(state_v)
                _, act_v = torch.max(q_vals_v, dim=1)
                action = act_v
        else:
            action = 0

        # do step in the environment
        new_state, reward, is_done = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, new_state, is_done)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, next_states, dones = batch

    states_v = torch.from_numpy(states).float().to(device)
    next_states_v = torch.from_numpy(next_states).float().to(device)
    actions_v = torch.tensor(actions).long().to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def calc_loss_double_dqn(batch, net, tgt_net, device="cpu", double=True):
    states, actions, rewards, next_states, dones = batch

    states_v = torch.from_numpy(states).float().to(device)
    actions_v = torch.tensor(actions).long().to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_vals = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        #next_states_v = torch.tensor(next_states).float().to(device)
        next_states_v = torch.from_numpy(next_states).float().to(device)
        if double:
            next_state_acts = net(next_states_v).max(1)[1]
            next_state_acts = next_state_acts.unsqueeze(-1)
            next_state_vals = tgt_net(next_states_v).gather(
                1, next_state_acts).squeeze(-1)
        else:
            next_state_vals = tgt_net(next_states_v).max(1)[0]

        next_state_vals[done_mask] = 0.0
        exp_sa_vals = next_state_vals.detach() * GAMMA + rewards_v
    return nn.MSELoss()(state_action_vals, exp_sa_vals)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    directory = os.path.dirname(os.getcwd())
    run_directory = os.path.join(directory, 'run/experiment_1')
    writer = SummaryWriter(run_directory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = environ.Env(dailydose_max=3.5, dose_step=0.5, tot_dose_max=60, num_fractions=40,
                      numba=True, state_space="Continuous")

    net = dqlmodel.SimpleNoisyFFDQN(env.observation.statespace.n, env.action_space.n).to(device)
    tgt_net = dqlmodel.SimpleNoisyFFDQN(env.observation.statespace.n, env.action_space.n).to(device)

    buffer = ReplayBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    epochs_idx = 0
    ts_epoch = 0
    ts = time.time()
    best_m_reward = None

    for i in range(NUM_EPISODES + 1):
        epochs_idx += 1
        #epsilon = max(EPSILON_END, EPSILON_START - epochs_idx / EPSILON_DECAY)

        reward = agent.play_step(net, epsilon=0, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (epochs_idx - ts_epoch) / (time.time() - ts)
            ts_epoch = epochs_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, reward %.3f, "
                  "eps %.2f, speed %.2f epochs/s" % (
                      epochs_idx, len(total_rewards), m_reward, epsilon,
                      speed
                  ))
            writer.add_scalar("epsilon", epsilon, epochs_idx)
            writer.add_scalar("speed", speed, epochs_idx)
            writer.add_scalar("reward_100", m_reward, epochs_idx)
            writer.add_scalar("reward", reward, epochs_idx)

            if best_m_reward is None or best_m_reward < m_reward:
                directory = os.path.dirname(os.getcwd())
                directory = os.path.join(directory, 'net')
                file_path = os.path.join(
                    directory, "-best_%.0f.dat" % m_reward)
                torch.save(net.state_dict(), file_path)
                if best_m_reward is not None:
                    print(f"Best reward updated {best_m_reward} -> {m_reward}")
                best_m_reward = m_reward

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if epochs_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        #loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t = calc_loss_double_dqn(batch, net, tgt_net, device=device, double = False)
        loss_t.backward()
        optimizer.step()
        writer.add_scalar('Train/Loss', loss_t.item(), epochs_idx)
    writer.close()
