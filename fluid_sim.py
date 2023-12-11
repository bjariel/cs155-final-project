# CS155 Final Project
# Authors: Bella Jariel, Kate Riggs, Luke Summers

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy as sp

class AnimatedScatter(object):
    def __init__(self, numpoints=30, disp=10):
        self.numpoints = numpoints
        self.disp = disp
        self.s = 0.016
        self.data = {
            'pos' : np.random.uniform(1.0, self.disp - 1.0, (self.numpoints, 2)),
            'vel' : np.zeros((self.numpoints, 2)),
            'acc' : np.zeros((self.numpoints, 2)),
            'mass' : np.ones(self.numpoints)
        }
        for i in range(self.numpoints):
            self.data['vel'][i] = self.get_vel(self.data['pos'][i, 0], self.data['pos'][i, 1])
        self.fig, self.ax = plt.subplots()
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5,
                                          init_func=self.setup_plot, blit=True)

    def kernel(self, x, y, h):
        # From https://philip-mocz.medium.com/create-your-own-smoothed-particle-hydrodynamics-simulation-with-python-76e1cec505f1
        # Gaussian Smooothing Kernel
        r = np.sqrt(x**2 + y**2)

        w = (1.0 / (h*np.sqrt(np.pi)))**3 * np.exp( -r**2 / h**2)

        return w

    def gradW(self, x, y, h):
        # From https://philip-mocz.medium.com/create-your-own-smoothed-particle-hydrodynamics-simulation-with-python-76e1cec505f1

        r = np.sqrt(x**2 + y**2)

        n = -2 * np.exp( -r**2 / h**2) / h**5 / (np.pi)**(3/2)
        wx = n * x
        wy = n * y

        return wx, wy

    def getDensities(self, r, h):
        densities = np.zeros(self.numpoints)
        for i in range(self.numpoints):
            total = 0
            for j in range(self.numpoints):
                if i == j:
                    continue
                r_ij = self.data['pos'][i] - self.data['pos'][j]
                total += self.kernel(r_ij[0], r_ij[1], h) * self.data['mass'][j]
            densities[i] = total
        return densities

    def getViscosities(self, densities, r, h, mu):
        viscousities = np.zeros((self.numpoints, 2))
        for i in range(self.numpoints):
            total = np.zeros(2)
            for j in range(self.numpoints):
                if i == j:
                    continue
                r_ij = self.data['pos'][i] - self.data['pos'][j]
                if densities[j] == 0:
                    continue
                total[0] += mu * self.data['vel'][j][0] * self.gradW(r_ij[0], r_ij[1], h)[0]**2 * self.data['mass'][j] / densities[j]
                total[1] += mu * self.data['vel'][j][1] * self.gradW(r_ij[0], r_ij[1], h)[1]**2 * self.data['mass'][j] / densities[j]
            viscousities[i] = total
        return viscousities

    def getPressures(self, densities, h, k, n):
        pressures = np.zeros(self.numpoints)
        for i in range(self.numpoints):
            pressures[i] = k * densities[i]**(1 + 1/n)
        return pressures

    def getAcc(self, pressures, densities, r, h, s):
        # Initialize acceleration arrays
        acc = np.zeros((self.numpoints, 2))
        for i in range(self.numpoints):
            for j in range(self.numpoints):
                # Calculate pairwise distances
                dx = self.data['pos'][i][0] - self.data['pos'][j][0]
                dy = self.data['pos'][i][1] - self.data['pos'][j][1]
                # Calculate gradient of W
                dWx, dWy = self.gradW(dx, dy, h)

                # Calculate pressure contribution to acceleration
                if densities[j] == 0:
                    continue
                acc[i][0] -= self.data['mass'][j] * pressures[i] / densities[i]**2 + pressures[j] / densities[j]**2 * dWx
                acc[i][1] -= self.data['mass'][j] * pressures[i] / densities[i]**2 + pressures[j] / densities[j]**2 * dWy
        return acc

    def smooth(self, r, h, s):
        # Find density at each point
        densities = self.getDensities(r, h)

        # Viscosity
        coeff_visc = 0.1
        viscousities = self.getViscosities(densities, r, h, coeff_visc)

        # Calculate pressure
        state_constant = 0.5
        polytropic_index = 0.5
        pressures = self.getPressures(densities, h, state_constant, polytropic_index)

        # Calculate acceleration due to pressure
        acc = self.getAcc(pressures, densities, r, h, s)
        return acc


    def vec_field(self, position):
        ## to_center = np.array([self.disp/2, self.disp/2]) - position
        X = position[0]
        Y = position[1]
        center = np.array([5, 5])
        U = -(Y - center[1])
        V = X - center[0]
        direction = [U, V]
        ## Update vector field, as of now just points out from center
        ## direction = [to_center[0], to_center[1]]
        return direction / np.linalg.norm(direction)

    def setup_plot(self):
        colors = np.random.rand(30)
        self.scat = self.ax.scatter(self.data['pos'][:, 0], self.data['pos'][:, 1], s=5000, c=colors)
        self.ax.axis([0, 10, 0, 10])
        return self.scat,

    def get_vel(self, x, y):
        val = np.sin(2 * x * y)
        return [val, val]

    def step(self, frame_number):
        sph_acc = self.smooth(self.data['pos'], 0.5, self.s)
        for n in range(self.numpoints):
            # self.data['vel'][n] = self.vec_field(self.data['pos'][n], self.s * frame_number / self.data['mass'][n])
            if self.data['pos'][n][0] < 0:
                self.data['pos'][n] = 0
            elif self.data['pos'][n][1] < 0:
                self.data['pos'][n] = 0
            elif self.data['pos'][n][0] > self.disp:
                self.data['pos'][n] = self.disp
            elif self.data['pos'][n][1] > self.disp:
                self.data['pos'][n] = self.disp
            else:
                self.data['pos'][n] += ((self.data['vel'][n] * self.s) * 0.5) + (self.vec_field(self.data['pos'][n]) * self.s) 
                self.data['vel'][n] += self.data['acc'][n] * self.s
                self.data['vel'][n] += sph_acc[n] * self.s
        return self.data['pos']

    def update(self, i):
        xy = self.step(i)
        self.scat.set_offsets(xy)
        return self.scat,


if __name__ == '__main__':
    a = AnimatedScatter()
    plt.show()
    #a.ani.save('animation.mp4', writer='ffmpeg', fps=10, extra_args=['-vcodec', 'libx264'], savefig_kwargs={'pad_inches':0}, dpi=300)