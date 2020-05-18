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
