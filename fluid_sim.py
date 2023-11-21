# CS155 Final Project
# Authors: Bella Jariel, Kate Riggs, Luke Summers

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class AnimatedScatter(object):
    def __init__(self, numpoints=10, disp=10):
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

    def setup_plot(self):
        self.scat = self.ax.scatter(self.data['pos'][:, 0], self.data['pos'][:, 1])
        self.ax.axis([-10, 10, -10, 10])
        return self.scat,

    def get_vel(self, x, y):
        val = np.sin(2 * x * y)
        return [val, val]

    def step(self):
        for n in range(self.numpoints):
            self.data['pos'][n] += self.data['vel'][n] * self.s
            self.data['vel'][n] += self.data['acc'][n] * self.s
        return self.data['pos']

    def update(self, i):
        xy = self.step()
        print(xy)
        self.scat.set_offsets(xy)
        return self.scat,


if __name__ == '__main__':
    a = AnimatedScatter()
    plt.show()