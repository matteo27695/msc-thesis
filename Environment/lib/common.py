import numpy as np


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
