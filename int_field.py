import numpy as np
import matplotlib.pyplot as plt
from pynput.mouse import Listener
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


# class InteractiveVectorField:
#     def __init__(self):
#         self.fig, self.ax = plt.subplots()

#         self.vector_field = None
#         self.quiver = None
#         self.particles = None
#         self.last_mouse_position = None

#         self.fig.canvas.mpl_connect('motion_notify_event', self.interact)

#     def vec_field(self):
#         x = np.linspace(-2, 2, 20)
#         y = np.linspace(-2, 2, 20)
#         X, Y = np.meshgrid(x, y)
#         U = np.sin(X) * np.cos(Y)
#         V = np.sin(Y) * np.cos(X)

#         return X, Y, U, V

#     def plot(self):
#         # https://www.geeksforgeeks.org/quiver-plot-in-matplotlib/#

#         self.vector_field = self.vec_field()
#         self.quiver = self.ax.quiver(self.vector_field[0], self.vector_field[1],
#                                      self.vector_field[2], self.vector_field[3],
#                                      scale=20)
#         self.ax.axis('equal')

#         self.particles = np.random.rand(10, 2) * 4 - 2 
#         self.particle_velocities = np.zeros((10, 2))
#         self.particle_dots, = self.ax.plot([], [], 'ro')

#         self.ani = animation.FuncAnimation(self.fig, self.animate, init_func=self.init_anim,
#                                            frames=200, interval=50, blit=True)
#         plt.show()

#     def init_anim(self):
#         return self.particle_dots,

#     def animate(self, frame):
#         if self.vector_field is not None:
#             X, Y, U, V = self.vector_field

#             for i, particle in enumerate(self.particles):
#                 x, y = particle
#                 row = int((x + 2) / 4 * (len(X[0]) - 1)) 
#                 col = int((y + 2) / 4 * (len(Y) - 1))

#                 self.particle_velocities[i, 0] = U[row, col]
#                 self.particle_velocities[i, 1] = V[row, col]

#                 dx = 0.05 * self.particle_velocities[i, 0]
#                 dy = 0.05 * self.particle_velocities[i, 1]

# ## check if pos is outside boundary and force back in rather than not moving outside
# ## check what the pos are when they disappear

# ## USE NEAREST NEIGHBOR SEARCH if we want > 100 particles
# ## look into int techniquies like midpoint or inverted euler for viscosity

#                 if row + dx < len(X[0]) >= 0 and col + dy < len(Y) >= 0:
#                     particle[0] += dx
#                     particle[1] += dy

#             self.particle_dots.set_data(self.particles[:, 0], self.particles[:, 1])
#             return self.particle_dots,

#     def interact(self, event):
#         # https://matplotlib.org/stable/users/explain/figure/event_handling.html
        
#         if event.xdata and event.ydata and event.button == 1:
#             if self.last_mouse_position is None:
#                 self.last_mouse_position = (event.xdata, event.ydata)
#                 return

#             delta_x = event.xdata - self.last_mouse_position[0]
#             delta_y = event.ydata - self.last_mouse_position[1]

#             X, Y, U, V = self.vector_field
#             U += delta_x
#             V += delta_y

#             self.vector_field = X, Y, U, V

#             self.quiver.remove()
#             self.quiver = self.ax.quiver(X, Y, U, V, scale=20)
#             self.last_mouse_position = (event.xdata, event.ydata)
#             plt.draw()

#     def run(self):
#         self.plot()

# if __name__ == "__main__":
#     interactive_field = InteractiveVectorField()
#     interactive_field.run()


class InteractiveParticles:
    def __init__(self):
        self.particles = np.random.rand(10, 2) * 4 - 2  # 10 particles within [-2, 2] range
        self.vector_field = self.create_vector_field()
        self.is_mouse_pressed = False

    def create_vector_field(self):
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        X, Y = np.meshgrid(x, y)
        U = np.sin(X) * np.cos(Y)
        V = np.sin(Y) * np.cos(X)

        return X, Y, U, V

    def update_particles(self):
        # Update particle velocities based on the current vector field
        row = ((self.particles[:, 0] + 2) / 4 * (len(self.vector_field[0][0]) - 1)).astype(int)
        col = ((self.particles[:, 1] + 2) / 4 * (len(self.vector_field[1]) - 1)).astype(int)
        self.particle_velocities = np.array([self.vector_field[2][row, col], self.vector_field[3][row, col]]).T

        # Update particle positions based on their velocities
        self.particles += self.particle_velocities * 0.05  # Adjust the factor for velocity magnitude

        # Ensure particles stay within bounds [-2, 2]
        self.particles = np.clip(self.particles, -2, 2)

    def modify_vector_field(self, delta_x, delta_y):
        # Update vector field based on mouse movement
        X, Y, U, V = self.vector_field
        U += delta_x
        V += delta_y

        return X, Y, U, V

    def on_click(self, x, y, button, pressed):
        if button == button.left:
            self.is_mouse_pressed = pressed

    def on_move(self, x, y):
        if self.is_mouse_pressed:
            delta_x = x / 100 - 2  # Normalize to [-2, 2]
            delta_y = y / 100 - 2

            # Update the vector field based on mouse movement
            self.vector_field = self.modify_vector_field(delta_x, delta_y)

    def simulate(self):
        fig, ax = plt.subplots()
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_title('Particles')

        # Start listening to mouse events
        with Listener(on_click=self.on_click, on_move=self.on_move) as listener:
            ani = FuncAnimation(fig, self.update_and_plot_particles, frames=200, interval=50, blit=False)
            plt.show()

    def update_and_plot_particles(self, frame):
        self.update_particles()
        plt.clf()
        plt.plot(self.particles[:, 0], self.particles[:, 1], 'ro')

if __name__ == "__main__":
    interactive_particles = InteractiveParticles()
    interactive_particles.simulate()
