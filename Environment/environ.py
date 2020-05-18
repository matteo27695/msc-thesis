import random

import numpy as np
from numba import njit
from scipy.integrate import odeint


@njit
def numba_model(z, t, u, a, b, c, alpha, ratio, gamma):
    x = z[0]
    y = z[1]
    dxdt = a * x * np.log(b / (x + c)) - alpha * (1 + (2 / ratio) * y) * u * x
    dydt = u - gamma * y**2
    dzdt = [dxdt, dydt]
    return dzdt


class OdeModel:
    def __init__(self, a, alpha, b, z0=[1, 0], c=0, ratio=10, gamma=71, days=1, timestep="minuto"):
        if timestep == "minuto":  # Timestep of the ode solver
            resolution = 1440  # Daily resolution for the ode solver
        elif timestep == "secondo":
            resolution = 86400
        self.days = days
        self.nt = self.days * resolution + 1  # Time points for simulation
        self.t = np.linspace(0, self.days, self.nt)
        self.z0 = z0

        self.a = a
        self.alpha = alpha
        self.b = z0[0] * b
        self.c = c
        self.ratio = ratio
        self.gamma = gamma

    def reset(self, z0):
        self.z0 = z0

    def model(self, z, t, u):
        x = z[0]
        y = z[1]
        dxdt = self.a * x * np.log(self.b / (x + self.c)) - \
            self.alpha * (1 + (2 / self.ratio) * y) * u * x
        dydt = u - self.gamma * y**2
        dzdt = [dxdt, dydt]
        return dzdt

    def solve(self, ut, z0, numba):
        self.x = np.empty_like(self.t)
        self.y = np.empty_like(self.t)

        z0 = z0
        # record initial conditions
        self.x[0] = z0[0]
        self.y[0] = z0[1]

        if numba:
            for i in range(0, self.nt - 1):
                # span for next time step
                tspan = [self.t[i], self.t[i + 1]]
                z = odeint(numba_model, z0, tspan, args=(
                    ut[i], self.a, self.b, self.c, self.alpha, self.ratio, self.gamma))  # solve for next step
                # store solution for plotting
                self.x[i + 1] = z[-1][0]
                self.y[i + 1] = z[-1][1]
                # next initial condition
                z0 = z[-1]
        else:
            for i in range(0, self.nt - 1):
                # span for next time step
                tspan = [self.t[i], self.t[i + 1]]
                z = odeint(self.model, z0, tspan, args=(
                    ut[i],))  # solve for next step
                # store solution for plotting
                self.x[i + 1] = z[-1][0]
                self.y[i + 1] = z[-1][1]
                # next initial condition
                z0 = z[-1]


class RFDynamicModel:
    def __init__(self, numba):
        self.numba = numba
        self.model = [OdeModel(a=0.15, alpha=np.random.normal(1.43, 1.43 * 5 / 100), b=2, gamma=43),
                      OdeModel(a=0.15, alpha=np.random.normal(0.15, 0.15 * 5 / 100), b=1, ratio=3)]
        # self.model[0] tumor dynamical evolution
        # self.model[1] normal cell dynamical evolution

    def reset(self):
        self.model = [model.reset([1, 0]) for model in self.model]

    def solve(self, z0, ut):
        for i in range(len(self.model)):
            self.model[i].solve(ut, z0[i], self.numba)
        next_state = [[model.x, model.y] for model in self.model]
        return next_state  # next_state value


class Action:
    @staticmethod
    def inputode(daily_dose, rad_day, days=1, t_radio=25, timestep="minuto"):
        tot_dose = daily_dose  # Total dosage of the daily fraction
        doserate_min = tot_dose / t_radio  # (Gy/min)
        doserate = doserate_min * (24 * 60)  # (Gy/die)
        if timestep == "minuto":  # Timestep of the ode solver
            resolution = 1440  # Daily resolution for the ode solver
            t_radio = t_radio  # Duration in minutes of the radiotherapy, it is the time in minutes to administer the daily dose
        elif timestep == "secondo":
            resolution = 86400
            # Duration in seconds of the radiotherapy, it is the time in minutes to administer the daily dose
            t_radio = t_radio * 60

        days = days       # Duration in days of radiotherapy treatment

        nt = days * resolution + 1  # Time points for ode simulation
        t = np.linspace(0, days, nt)

        day_res = np.linspace(0, 1, resolution + 1)  # Daily resolution

        u_noradioday = np.zeros(len(day_res))  # Input of not radiation day
        u_radioday = np.zeros(len(day_res))  # Input of radiation day
        u_radioday[0:t_radio + 1] = 1

        ut_min = u_radioday * doserate_min
        ut = u_radioday * doserate

        if rad_day:
            return ut
        else:
            return u_noradioday

    @staticmethod
    def radio_day(day):
        return True if day != 6 and day != 7 else False


class Observation:
    def __init__(self, tot_dose_max, tot_fractions, state_space):
        self.tumor_lastvalue = 1
        self.healthy_lastvalue = 1

        self.tumor_evolution = [self.tumor_lastvalue]
        self.healthy_evolution = [self.healthy_lastvalue]

        self.gammatumor_lastvalue = 0
        self.gammahealthy_lastvalue = 0

        self.tot_dose = 0
        self.fractions = 0

        self.statespace_type = state_space
        if self.statespace_type == "Discrete":
            self.statespace = DiscreteStateSpace(tot_dose_max, tot_fractions)
        else:
            self.statespace = ContinuousStateSpace([self.tumor_lastvalue, self.healthy_lastvalue,
                                                    self.tot_dose, self.fractions])

        self.z0 = [[self.tumor_lastvalue, self.gammatumor_lastvalue],
                   [self.healthy_lastvalue, self.gammahealthy_lastvalue]]

    def update(self, next_state):
        self.tumor_evolution.extend(next_state[0][0][1:])
        self.healthy_evolution.extend(next_state[1][0][1:])

        self.tumor_lastvalue = next_state[0][0][-1]
        self.healthy_lastvalue = next_state[1][0][-1]

        self.gammatumor_lastvalue = next_state[0][1][-1]
        self.gammahealthy_lastvalue = next_state[1][1][-1]

        # self.tot_dose e self.fractions aggiornati esternamente
        self.statespace.update(self.tumor_lastvalue, self.healthy_lastvalue,
                               self.tot_dose, self.fractions)

        self.z0 = [[self.tumor_lastvalue, self.gammatumor_lastvalue],
                   [self.healthy_lastvalue, self.gammahealthy_lastvalue]]

    def reset(self):
        self.tumor_lastvalue = 1
        self.healthy_lastvalue = 1

        self.tumor_evolution = [self.tumor_lastvalue]
        self.healthy_evolution = [self.healthy_lastvalue]

        self.gammatumor_lastvalue = 0
        self.gammahealthy_lastvalue = 0

        self.tot_dose = 0
        self.fractions = 0

        self.statespace.reset()  # VERIFICA#
        self.z0 = [[self.tumor_lastvalue, self.gammatumor_lastvalue],
                   [self.healthy_lastvalue, self.gammahealthy_lastvalue]]


class ContinuousStateSpace:
    def __init__(self, var_values):
        self.n = len(var_values)

    def update(self, last_tumorvalue, last_healthvalue, tot_dosage, fractions):
        self.stateval = np.array(
            [last_tumorvalue, last_healthvalue, tot_dosage, fractions])

    def state(self):
        return self.stateval

    def reset(self):
        self.stateval = np.array([1, 1, 0, 0])


class DiscreteStateSpace:
    def __init__(self, tot_dose_max, tot_fractions, resolution=20):
        n = 2  # group size
        m = 1  # overlap size

        states = np.linspace(0.65, 1, 1 * 30 + 1)
        self.statetumor = [states.tolist()[i:i + n]
                           for i in range(0, len(states.tolist()), n - m)]
        self.statetumor.remove(self.statetumor[-1])

        states = np.linspace(0.8, 1, 1 * resolution + 1)
        self.statehealth = [states.tolist()[i:i + n]
                            for i in range(0, len(states.tolist()), n - m)]
        self.statehealth.remove(self.statehealth[-1])

        dosestates = np.linspace(0, tot_dose_max, 1 * resolution + 1)
        self.statetotdose = [dosestates.tolist()[i:i + n]
                             for i in range(0, len(dosestates.tolist()), n - m)]
        self.statetotdose.remove(self.statetotdose[-1])

        self.tumor_index = resolution - 1
        self.healthy_index = resolution - 1
        self.tot_dosage_index = 0
        self.fractions_index = 0

        self.n = len(self.statetumor), len(self.statehealth), len(
            self.statetotdose), tot_fractions

    def update(self, last_tumorvalue, last_healthvalue, tot_dosage, fractions):
        if last_tumorvalue < self.statetumor[0][1]:
            self.tumor_index = 0
        if last_tumorvalue > self.statetumor[len(self.statetumor) - 1][1]:
            self.tumor_index = len(self.statetumor) - 1
        else:
            for i in range(len(self.statetumor)):
                if self.statetumor[i][0] <= last_tumorvalue <= self.statetumor[i][1]:
                    self.tumor_index = i
                    break
        if last_healthvalue < self.statehealth[0][1]:
            self.healthy_index = 0
        if last_healthvalue > self.statehealth[len(self.statehealth) - 1][1]:
            self.healthy_index = len(self.statehealth) - 1
        else:
            for i in range(len(self.statehealth)):
                if self.statehealth[i][0] <= last_healthvalue <= self.statehealth[i][1]:
                    self.healthy_index = i
                    break

        if tot_dosage > self.statetotdose[len(self.statetotdose) - 1][1]:
            self.tot_dosage_index = len(self.statetotdose) - 1
        else:
            for i in range(len(self.statetotdose)):
                if self.statetotdose[i][0] <= tot_dosage <= self.statetotdose[i][1]:
                    self.tot_dosage_index = i
                    break

        self.fractions_index = fractions - 1

    def state(self):
        return self.tumor_index, self.healthy_index, self.tot_dosage_index, self.fractions_index

    def reset(self):
        self.tumor_index = resolution - 1
        self.healthy_index = resolution - 1

        self.tot_dosage_index = 0
        self.fractions_index = 0


class ActionSpace:
    def __init__(self, dose_max, dose_step):
        self.space = np.arange(0, dose_max + dose_step, dose_step)
        self.n = len(self.space)

    def value(self, index):
        return self.space[index]

    def sample(self):
        return random.randint(0, self.n - 1)


class Env:
    def __init__(self, dailydose_max, dose_step, tot_dose_max, num_fractions, numba, state_space="Continuous"):
        self.model = RFDynamicModel(numba)

        self.num_fractions = num_fractions

        self.action_space = ActionSpace(dailydose_max, dose_step)
        self.observation = Observation(
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
            return (1-tumor_lastvalue)

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
        rad_day = Action.radio_day(self.daily_week)
        if rad_day:
            self.observation.fractions += 1
            self.observation.tot_dose += daily_dose
        ut = Action.inputode(daily_dose, rad_day)  # Input vector to ODEs model
        next_state = self.model.solve(z0, ut)

        reward = Env.reward(next_state)  # Create Function
        #reward = next_state[0][0][-1]

        # Questa funzione deve venire per forza dopo all'aggiornamento di tot_dose
        self.observation.update(next_state)

        if self.observation.tot_dose > self.tot_dose_max or self.observation.fractions == self.num_fractions or self.observation.tumor_lastvalue < 0.00001:
            # if self.observation.tot_dose > self.tot_dose_max or self.days > 6:
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
        # return self.observation
        return self.observation.statespace.state()

        # ACTION INDEX QUANDO è WEEKEND, CONTROLLO INTERNO A ENV MA Q_TABLE AGGIORNATA.
