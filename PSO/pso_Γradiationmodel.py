# ------------------------------------------------------------------------------+
#
#   Matteo Tortora
#   Particle Swarm Optimization (PSO) algorithm to estimate parameters in
#   Ordinary Differential Equations Modeling.
#   February, 2020
#
# ------------------------------------------------------------------------------+

import math
import multiprocessing
from datetime import date, datetime, timedelta
from random import random, uniform

import numpy as np
import psutil
from IPython import display
from scipy.integrate import odeint


class OdeModel:
    def __init__(self, days, ut, z0=[1, 0], ratio=10, gamma=71, b=1000, c=0, timestep="minuto"):
        if timestep == "minuto":  # Timestep of the ode solver
            resolution = 1440  # Daily resolution for the ode solver
        elif timestep == "secondo":
            resolution = 86400
        self.days = days
        self.nt = self.days * resolution + 1  # Time points for simulation
        self.t = np.linspace(0, self.days, self.nt)
        self.z0 = z0
        self.ut = ut
        self.b = b
        self.c = c
        self.ratio = ratio
        self.gamma = gamma

    def param(self, a, alpha, b=None, c=None, ratio=None, gamma=None):
        self.a = a
        self.alpha = alpha
        self.b = b if b is not None else self.b
        self.c = c if c is not None else self.c
        self.ratio = ratio if ratio is not None else self.ratio
        self.gamma = gamma if gamma is not None else self.gamma

    def model(self, z, t, u):
        x = z[0]
        y = z[1]
        dxdt = self.a * x * np.log(self.b / (x + self.c)) - \
            self.alpha * (1 + (2 / self.ratio) * y) * u * x
        dydt = u - self.gamma * y**2
        dzdt = [dxdt, dydt]
        return dzdt

    def solve(self):
        self.x = np.empty_like(self.t)
        self.y = np.empty_like(self.t)

        z0 = self.z0
        # record initial conditions
        self.x[0] = z0[0]
        self.y[0] = z0[1]

        for i in range(0, self.nt - 1):  # MODIFICA range(1,self.nt)
            # span for next time step #mod: tspan = [self.t[i-1],self.t[i]]
            tspan = [self.t[i], self.t[i + 1]]
            z = odeint(self.model, z0, tspan, args=(
                self.ut[i],))  # solve for next step
            # store solution for plotting
            self.x[i + 1] = z[-1][0]
            self.y[i + 1] = z[-1][1]
            # next initial condition
            z0 = z[-1]


class RegressionMetrics:
    @staticmethod
    def metric_rss(observed, estimated):
        # Mean Absolute Error
        rss = np.sum(np.square(observed - estimated))
        return rss

    @staticmethod
    def metric_mse(observed, estimated):
        # Mean Squared Error
        mse = np.mean((np.square(observed - estimated)))
        return mse

    @staticmethod
    def metric_msep(observed, estimated):
        # Mean Squared Error of Prediction
        msep = np.mean(
            np.divide(np.square(observed - estimated), np.square(observed)))
        return msep

    @classmethod
    def metric_rmse(cls, observed, estimated):
        # Root Mean Square Error
        rmse = np.square(cls.metric_mse(observed, estimated))
        return rmse

    @classmethod
    def metric_rmsep(cls, observed, estimated):
        # Root Mean Square Error of Prediction
        rmsep = np.square(cls.metric_msep(observed, estimated))
        return rmsep

    @staticmethod
    def metric_mae(observed, estimated):
        # Mean Absolute Error
        mae = np.mean(np.abs(observed - estimated))
        return mae

    @staticmethod
    def metric_mape(observed, estimated):
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs(np.divide((observed - estimated), observed)))
        return mape

    @staticmethod
    def metric_rse(observed, estimated):
        # Relative Squared Error
        rse = np.divide(np.sum(np.square(observed - estimated)),
                        np.sum(np.square(observed - np.mean(observed))))
        return rse

    @staticmethod
    def metric_rae(observed, estimated):
        # Relative Absolute Error
        rae = np.divide(np.sum(np.abs(observed - estimated)),
                        np.sum(np.abs(observed - np.mean(observed))))
        return rae

    @staticmethod
    def metric_rss(observed, estimated):
        # Residual Sum of Squares
        rss = np.sum(np.square(observed - estimated))
        return rss

    @classmethod
    def metric_r2(cls, observed, estimated):
        r2 = 1 - cls.metric_rse(observed, estimated)
        return r2


class PrepareData:
    def inputode(start_date, end_date, tot_dose, t_radio=25, timestep="minuto"):
        tot_dose = tot_dose  # Total dosage of the daily fraction
        doserate_min = tot_dose / t_radio  # (Gy/min)
        doserate = doserate_min * (24 * 60)  # (Gy/die)
        if timestep == "minuto":  # Timestep of the ode solver
            resolution = 1440  # Daily resolution for the ode solver
            t_radio = t_radio  # Duration in minutes of the radiotherapy, it is the time in minutes to administer the daily dose
        elif timestep == "secondo":
            resolution = 86400
            # Duration in seconds of the radiotherapy, it is the time in minutes to administer the daily dose
            t_radio = t_radio * 60
        # Start date of the radiotherapy
        sdate = datetime.strptime(start_date, '%d-%m-%y').date()
        # End date of the radiotherapy
        edate = datetime.strptime(end_date, '%d-%m-%y').date()

        delta = edate - sdate
        days = delta.days + 1       # Duration in days of radiotherapy treatment

        nt = days * resolution + 1  # Time points for ode simulation
        t = np.linspace(0, days, nt)

        day_res = np.linspace(0, 1, resolution + 1)  # Daily resolution
        u_noradioday = np.zeros(len(day_res))  # Input of not radiation day
        u_radioday = np.zeros(len(day_res))  # Input of radiation day
        u_radioday[0:t_radio + 1] = 1

        ut = 0  # Input for ode simulation
        # Weekly day of the first day of therapy, if the first day is Monday then day = 1
        day = sdate.isoweekday()
        for i in range(0, days):
            if days == 1:
                if day == 6 or day == 7:
                    ut = u_noradioday
                    if day == 7:
                        day = 1
                    else:
                        day += 1
                else:
                    ut = u_radioday
            else:
                if i == 0:
                    if day == 6 or day == 7:
                        ut = u_noradioday[0:-1]
                        if day == 7:
                            day = 1
                        else:
                            day += 1
                    else:
                        ut = u_radioday[0:-1]
                        day += 1
                elif i == days - 1:
                    if day == 6 or day == 7:
                        ut = np.append(ut, u_noradioday)
                        if day == 7:
                            day = 1
                        else:
                            day += 1
                    else:
                        ut = np.append(ut, u_radioday)
                        day += 1
                else:
                    if day == 6 or day == 7:
                        ut = np.append(ut, u_noradioday[0:-1])
                        if day == 7:
                            day = 1
                        else:
                            day += 1
                    else:
                        ut = np.append(ut, u_radioday[0:-1])
                        day += 1
        ut_min = ut * doserate_min
        ut = ut * doserate
        return [ut, t, days, nt, ut_min]

    def realdata(start_date, end_date, dates, volumes, timestep="minuto"):
        if timestep == "minuto":  # Timestep of the ode solver
            resolution = 1440  # Daily resolution for the ode solver
        elif timestep == "secondo":
            resolution = 86400

        start_date = datetime.strptime(
            start_date, '%d-%m-%y').date()   # start date
        end_date = datetime.strptime(end_date, '%d-%m-%y').date()   # end date
        inputdata = np.zeros((end_date - start_date).days + 1)
        time = np.zeros(len(volumes))
        i = 0
        for date, volume in zip(dates, volumes):
            date = datetime.strptime(date, '%d-%m-%y').date()
            delta = (date - start_date).days
            time[i] = delta
            inputdata[delta] = volume
            i += 1

        zero = np.zeros(resolution)
        for i in range(0, len(inputdata)):
            if i == 0:
                outdata = inputdata[i]
                outdata = np.append(outdata, zero[0:-1])
            elif i == len(inputdata) - 1:
                outdata = np.append(outdata, inputdata[i])
                outdata = np.append(outdata, zero)
            else:
                outdata = np.append(outdata, inputdata[i])
                outdata = np.append(outdata, zero[0:-1])
        return [outdata, time]


class Particle:
    def __init__(self, dim):
        self.pos = np.array([0.01217*uniform(0.9, 1.1), 0.012*uniform(0., 1.1)])
        # self.pos = np.array(
        #    [0.0155 * uniform(0.9, 1), 0.013 * uniform(0.9, 1)])
        self.vel = np.zeros(dim)
        self.pbest_pos = self.pos
        self.pbest_val = np.inf

    def upd_pos(self):
        self.pos = self.pos + self.vel

    def upd_vel(self, w_inertia, w_cogn, w_soci, gbest_pos):
        inertia = w_inertia * self.vel
        bestind = w_soci * random() * (self.pbest_pos - self.pos)
        bestswa = w_cogn * random() * (gbest_pos - self.pos)
        self.vel = inertia + bestind + bestswa


class Swarm:
    def __init__(self, num_particles, n_iterations, dimspace, odemodel, realdata, mode='Parallel', tolerance=0.0001, w_inertia=0.5, w_cogn=0.8, w_soci=0.9):
        self.w_inertia = w_inertia
        self.w_cogn = w_cogn
        self.w_soci = w_soci
        self.tolerance = tolerance
        self.n_iterations = n_iterations
        self.dimspace = dimspace
        self.gbest_pos = np.empty(self.dimspace)
        self.gbest_val = [np.inf]
        self.num_particles = num_particles
        self.particles = []
        self.mode = mode
        self.model = odemodel
        #self.model.param(a = 0.015, alpha = 0.15)
        # self.model.solve()
        #self.realdata = self.model.y
        self.realdata = realdata

    def init_particles(self):
        self.particles = [Particle(self.dimspace)
                          for i in range(self.num_particles)]

    def fitness_radiation(self, particle):
        self.model.param(a=particle.pos[0], alpha=particle.pos[1])
        self.model.solve()
        estimated = self.model.x
        # return np.sum(np.square(self.realdata[self.realdata != 0] - estimated[self.realdata != 0]))
        return RegressionMetrics.metric_mape(self.realdata[self.realdata != 0], estimated[self.realdata != 0])

    def fitness(self, particle):
        self.model.param(a=particle.pos[0], alpha=particle.pos[1])
        self.model.solve()
        actualout = self.model.x
        return np.sum(np.square(np.asarray(self.realdata, dtype=None, order=None) - np.asarray(actualout, dtype=None, order=None)))

    def upd_pbest(self):
        for particle in self.particles:
            #fitness_value = self.fitness(particle)
            fitness_value = self.fitness_radiation(particle)
            if fitness_value < particle.pbest_val:
                particle.pbest_val = fitness_value
                particle.pbest_pos = particle.pos

    def upd_pbest_pool(self, particle):
        #fitness_value = self.fitness(particle)
        fitness_value = self.fitness_radiation(particle)
        if fitness_value < particle.pbest_val:
            particle.pbest_val = fitness_value
            particle.pbest_pos = particle.pos
        return particle

    def upd_gbest(self):
        expected_gbest_val = self.gbest_val[-1]
        for particle in self.particles:
            if particle.pbest_val < expected_gbest_val:
                expected_gbest_val = particle.pbest_val
                expected_gbest_pos = particle.pbest_pos
        if expected_gbest_val < self.gbest_val[-1]:
            if self.gbest_val[-1] == np.inf:
                self.gbest_val[-1] = expected_gbest_val
                self.gbest_pos = expected_gbest_pos
            else:
                self.gbest_val.append(expected_gbest_val)
                self.gbest_pos = expected_gbest_pos

    def move_particles(self):
        for particle in self.particles:
            particle.upd_vel(self.w_inertia, self.w_cogn,
                             self.w_soci, self.gbest_pos)
            particle.upd_pos()

    def move_particles_pool(self, particle):
        particle.upd_vel(self.w_inertia, self.w_cogn,
                         self.w_soci, self.gbest_pos)
        particle.upd_pos()
        return particle

    def run(self):
        if self.mode == 'Parallel':
            start_time = datetime.now()
            self.init_particles()
            num_cores = psutil.cpu_count(logical=False)
            i = 1
            old_len = 1
            pool = multiprocessing.Pool(num_cores)
            while i < self.n_iterations + 1:
                print(f'Iteration Number: {i}')
                print(f'RSS Value: {self.gbest_val[-1]}')
                display.clear_output(wait=True)
                self.particles = pool.map(self.upd_pbest_pool, [
                                          particle for particle in self.particles])
                self.upd_gbest()
                self.particles = pool.map(self.move_particles_pool, [
                                          particle for particle in self.particles])
                count = 0
                if len(self.gbest_val) > 2:
                    if old_len == len(self.gbest_val):
                        count += 1
                    else:
                        count = 0
                    len(self.gbest_val)
                    if abs(self.gbest_val[-1] - self.gbest_val[-2]) < self.tolerance or count > 5:
                        break
                old_len = len(self.gbest_val)
                i += 1
            pool.close()
            pool.join()
        elif self.mode == "Single":
            start_time = datetime.now()
            num_cores = 1
            self.init_particles()
            old_len = 1
            i = 1
            while i < self.n_iterations + 1:
                print(f'Iteration Number: {i}')
                print(f'RSS Value: {self.gbest_val[-1]}')
                display.clear_output(wait=True)
                self.upd_pbest()
                self.upd_gbest()
                self.move_particles()
                count = 0
                if len(self.gbest_val) > 2:
                    if old_len == len(self.gbest_val):
                        count += 1
                    else:
                        count = 0
                    len(self.gbest_val)
                    if abs(self.gbest_val[-1] - self.gbest_val[-2]) < self.tolerance or count > 5:
                        break
                old_len = len(self.gbest_val)
                i += 1
        self.run_time = (datetime.now() - start_time).total_seconds()
        print(f"Numbers of cores: {num_cores}")
        print(
            f"The best position is: {self.gbest_pos} with value: {self.gbest_val[-1]}, in iteration number: {i-1}")
        print(
            f"Time execution: {self.run_time} seconds")
