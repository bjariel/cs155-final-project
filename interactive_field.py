import matplotlib.pyplot as plt
import numpy as np

class InteractiveVectorField:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Interactive Vector Field')
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')

        self.vector_field = None
        self.quiver = None
        self.last_mouse_position = None

        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_drag)

    def create_vector_field(self):
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        X, Y = np.meshgrid(x, y)
        U = np.sin(X) * np.cos(Y)
        V = np.sin(Y) * np.cos(X)

        return X, Y, U, V

    def plot_vector_field(self):
        self.vector_field = self.create_vector_field()
        self.quiver = self.ax.quiver(self.vector_field[0], self.vector_field[1],
                                     self.vector_field[2], self.vector_field[3],
                                     scale=20)
        self.ax.axis('equal')
        plt.show()

    def on_mouse_drag(self, event):
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
        self.plot_vector_field()

if __name__ == "__main__":
    interactive_field = InteractiveVectorField()
    interactive_field.run()
