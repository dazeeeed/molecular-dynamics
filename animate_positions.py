# ==========================================
# Title:  Symulacja dynamiki molekularnej
# Author: 01141448
# Date:   04 Nov 2021
# ==========================================

import matplotlib

matplotlib.use('Qt5Agg')
# matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


def initialize_values():
    params = {}
    with open("parameters.txt", 'r') as f:
        lines = f.readlines()

    for line in lines:
        param_name, val = line.replace('\n', '').split('\t')
        params[param_name] = float(val)

    return params


def create_sphere(L):
    phi = np.linspace(0, 2 * np.pi, 100)
    theta = np.linspace(0, np.pi, 100)
    x_sphere = L * np.outer(np.cos(phi), np.sin(theta))
    y_sphere = L * np.outer(np.sin(phi), np.sin(theta))
    z_sphere = L * np.outer(np.ones(np.size(phi)), np.cos(theta))
    return x_sphere, y_sphere, z_sphere


def main():
    params = initialize_values()
    data = pd.read_csv(os.path.join('data', 'atom-positions.txt'), sep='\t', header=None)
    data = np.array(data)
    data = np.reshape(data, (-1, int(params.get('n')) ** 3, 4))  # frame, n-atom, atom parameters
    x = data[:, :, 0]  # frame, n-atom
    y = data[:, :, 1]
    z = data[:, :, 2]

    # sphere
    x_sphere, y_sphere, z_sphere = create_sphere(params.get('L'))

    def animate_scatters(iteration):
        scatter._offsets3d = (x[iteration, :], y[iteration, :], z[iteration, :])
        return scatter

    # Attaching 3D axis to the figure
    fig = plt.figure(figsize=(10, 10))  # should be 10,10
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # Initialize scatters
    scatter = ax.scatter(x[0, :], y[0, :], x[0, :],
                         c='red')  # c=x[0, :]**2+y[0, :]**2+z[0, :]**2, cmap='turbo', s=2000, alpha=0.8)  # real size is about 2000
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.1)

    # Number of iterations
    iterations = len(data)

    # Setting the axes properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.minorticks_on()

    # Provide starting angle for the view.
    ax.view_init(elev=-162, azim=-80)
    # ax.view_init(elev=-90, azim=0)

    ani = animation.FuncAnimation(fig, animate_scatters, iterations,
                                  interval=10, blit=False, repeat=True)

    save_animation = False
    if save_animation:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=-1, extra_args=['-vcodec', 'libx264'])
        ani.save('atom-movement.mp4', writer=writer)

    plt.show()


if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    main()
