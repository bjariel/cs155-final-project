import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class InteractiveVectorField:
    def __init__(self):
        self.fig, self.ax = plt.subplots()

        self.vector_field = None
        self.quiver = None
        self.particles = None
        self.last_mouse_position = None

        self.fig.canvas.mpl_connect('motion_notify_event', self.interact)

    def vec_field(self):
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        X, Y = np.meshgrid(x, y)
        U = np.sin(X) * np.cos(Y)
        V = np.sin(Y) * np.cos(X)

        return X, Y, U, V

    def plot(self):
        # https://www.geeksforgeeks.org/quiver-plot-in-matplotlib/#

        self.vector_field = self.vec_field()
        self.quiver = self.ax.quiver(self.vector_field[0], self.vector_field[1],
                                     self.vector_field[2], self.vector_field[3],
                                     scale=20)
        self.ax.axis('equal')

        self.particles = np.random.rand(10, 2) * 4 - 2 
        self.particle_velocities = np.zeros((10, 2))
        self.particle_dots, = self.ax.plot([], [], 'ro')

        self.ani = animation.FuncAnimation(self.fig, self.animate, init_func=self.init_anim,
                                           frames=200, interval=50, blit=True)
        plt.show()

    def init_anim(self):
        return self.particle_dots,

    def animate(self, frame):
        if self.vector_field is not None:
            X, Y, U, V = self.vector_field

            for idx, particle in enumerate(self.particles):
                x, y = particle
                row = int((x + 2) / 4 * (len(X[0]) - 1)) 
                col = int((y + 2) / 4 * (len(Y) - 1))

                self.particle_velocities[idx, 0] = U[row, col]
                self.particle_velocities[idx, 1] = V[row, col]

                dx = 0.05 * self.particle_velocities[idx, 0]
                dy = 0.05 * self.particle_velocities[idx, 1]

                if 0 <= row + dx < len(X[0]) and 0 <= col + dy < len(Y):
                    particle[0] += dx
                    particle[1] += dy

            self.particle_dots.set_data(self.particles[:, 0], self.particles[:, 1])
            return self.particle_dots,

    def interact(self, event):
        # https://matplotlib.org/stable/users/explain/figure/event_handling.html
        
        if event.xdata and event.ydata and event.button == 1:
            if self.last_mouse_position is None:
                self.last_mouse_position = (event.xdata, event.ydata)
                return

            delta_x = event.xdata - self.last_mouse_position[0]
            delta_y = event.ydata - self.last_mouse_position[1]

            X, Y, U, V = self.vector_field
            U += delta_x
            V += delta_y

            self.vector_field = X, Y, U, V

            self.quiver.remove()
            self.quiver = self.ax.quiver(X, Y, U, V, scale=20)
            self.last_mouse_position = (event.xdata, event.ydata)
            plt.draw()

    def run(self):
        self.plot()

if __name__ == "__main__":
    interactive_field = InteractiveVectorField()
    interactive_field.run()