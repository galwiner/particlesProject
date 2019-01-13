import matplotlib.pyplot as plt
import numpy as np
from particlesModule import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import animation, rc
from mpl_toolkits.mplot3d import Axes3D  # registers 3d perspective for scatter plot. nothing else


N = 100
T = 300
sim = simulator(particles=[particle(np.random.normal(-0.1, 0.1, 3), np.random.normal(0, 1, 3)) for p in range(N)],
                forceFunc=lambda r, p: -r - 0 * p, Tend=0.01)
(x, y, z) = sim.positions()

fig=plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.set_xlim3d([-1, 1])
ax.set_ylim3d([-1, 1])
ax.set_zlim3d([-1, 1])
ax.scatter(x, y, z)
t=[0]
def animate(i):
    #     y_i = F[i, ::3]
    ax.clear()
    # ax2.clear()
    sim.run()
    t.append(i)
    # avgp.append(sim.calc_avg_p())
    # avgr.append(sim.calc_avg_r())
    (x, y, z) = sim.positions()
    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([-1, 1])
    ax.scatter(x, y, z)
    # ax2.plot(t, avgp)
    # ax2.plot(t, avgr)


anim = animation.FuncAnimation(fig, animate, interval=100, frames=10)
plt.show()