from datetime import datetime

import numpy as np
from lib import common, odemodel


class Env:
    def __init__(self, dailydose_max, dose_step, tot_dose_max, num_fractions, numba, state_space="Continuous"):
        self.model = odemodel.RFDynamicModel(numba)

        self.num_fractions = num_fractions

        self.action_space = common.ActionSpace(dailydose_max, dose_step)
        self.observation = common.Observation(
            tot_dose_max, self.num_fractions, state_space)

        self.tot_dose_max = tot_dose_max
        self.daily_week = 1
        self.days = 0
        self.done = False

    @staticmethod
    def reward(next_state):
        tumor_lastvalue = next_state[0][0][-1]
        healthy_lastvalue = next_state[1][0][-1]
        if healthy_lastvalue < 0.85:
            return -1
        else:
            return (1 - tumor_lastvalue)

    def step(self, action_index):
        """
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (index): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
        """
        self.days += 1

        daily_dose = self.action_space.value(action_index)
        z0 = self.observation.z0  # ODEs model's initial conditions

        # If rad_day is True it's a radiation day in the therapy schedule.
        rad_day = common.Action.radio_day(self.daily_week)
        if rad_day:
            self.observation.fractions += 1
            self.observation.tot_dose += daily_dose
        # Input vector to ODEs model
        ut = common.Action.inputode(daily_dose, rad_day)
        next_state = self.model.solve(z0, ut)

        reward = Env.reward(next_state)  # Create Function

        # Questa funzione deve venire per forza dopo all'aggiornamento di tot_dose
        self.observation.update(next_state)

        if self.observation.tot_dose > self.tot_dose_max or self.observation.fractions == self.num_fractions or self.observation.tumor_lastvalue < 0.00001:
            self.done = True

        if self.daily_week != 7:
            self.daily_week += 1
        else:
            self.daily_week = 1

        return self.observation.statespace.state(), reward, self.done

    def reset(self):
        """
        When end of episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        """
        self.daily_week = 1
        self.days = 0

        self.done = False

        self.observation.reset()
        return self.observation.statespace.state()


if __name__ == '__main__':
    env = Env(dailydose_max=3, dose_step=0.5,
              tot_dose_max=60, num_fractions=40, numba=True)
    giorni = 30
    tot_reward = []

    start_time = datetime.now()
    for i in range(giorni):
        observation, reward, done = env.step(4)
        tot_reward.append(reward)
        if env.done:
            break
    print(
        f"Time execution: {(datetime.now() - start_time).total_seconds()} seconds")

    giorni = i + 1
    t = np.linspace(0, giorni, giorni * 1440 + 1)
    plt.plot(t, observation.tumor_evolution, "b", label='Tumor Evolution')
    plt.plot(t, observation.healthy_evolution,
             "r", label='Health Cell Evolution')
    plt.xlabel("Time (day)")
    plt.ylabel("% Shrinkage")
    plt.title("DRL Adaption Radiation Therapy")
    plt.legend()
