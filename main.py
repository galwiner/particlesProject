import numpy as np
from particlesModule import *
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # registers 3d perspective for scatter plot. nothing else

N = 10  # number of particles in the simulation

# generate a simulator object. the simulation has N particles with random position r and momentum p. both r and p are
# drawn from a normal distribution. TODO: draw momentum from a maxwell boltzmann distribution

sim = simulator(particles=[particle(np.random.normal(-0.1, 0.1, 3), np.random.normal(-1, 1, 3)) for p in range(N)],
                forceFunc=lambda r, p: -2 * r - 0.5 * p, Tend=0.01)
(x, y, z) = sim.positions()

# configure the plot
fig = plt.figure()
gs = GridSpec(4, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])  # XY projection
ax2 = fig.add_subplot(gs[0, 1])  # YZ projection
ax3 = fig.add_subplot(gs[0, 2])  # ZX projection
ax4 = fig.add_subplot(gs[1, :])  # line plot for the evolution of the average momentum
ax5 = fig.add_subplot(gs[2, :])  # line plot for the evolution of the average distance from the origin
ax6 = fig.add_subplot(gs[3, 1], projection='3d')  # 3d scatter plot of the particle positions
fig.suptitle('Particle simulation')

scat = ax1.scatter(x, y, animated=True)
scat2 = ax2.scatter(y, z, animated=True)
scat3 = ax3.scatter(z, x, animated=True)

com = sim.calc_com()  # calculate center of mass
com1 = ax1.plot(com[0], com[1], 'or-', animated=True)[0]
com2 = ax2.plot(com[1], com[2], 'or-', animated=True)[0]
com3 = ax3.plot(com[2], com[0], 'or-', animated=True)[0]
scat4, = ax6.plot(x, y, z, linestyle="", marker='o')
ax4.set_ylim(auto=True)
ax5.set_ylim(auto=True)
ax4.set_xlim(auto=True)
ax5.set_xlim(auto=True)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax2.set_xlabel('y')
ax2.set_ylabel('z')
ax3.set_xlabel('z')
ax3.set_ylabel('x')

t = [0]
avgp = [0]
avgr = [0]
pplot = ax4.plot(t, avgp, avgp, animated=True)[0]
ax4.set_ylabel('|p|')
rplot = ax5.plot(t, avgr, animated=True)[0]
ax5.set_ylabel('|r|')

# setting up the quiver plots on ax1,2,3
X = np.arange(-1, 1, 0.1)
Y = np.arange(-1, 1, 0.1)
U, V = np.meshgrid(X, Y)
meanfield1 = ax1.quiver(X, Y, U - com[0], V - com[1], scale=0.01)
meanfield2 = ax2.quiver(X, Y, U - com[1], V - com[2], scale=0.01)
meanfield3 = ax3.quiver(X, Y, U - com[2], V - com[0], scale=0.01)


# plt.tight_layout()


def animate(i):
    '''function generating the animation. it is automatically called repeatedly by FuncAnimation'''
    sim.run()  # run one simulation step, evolving particles positions
    t.append(t[-1] + 1)
    avgp.append(sim.calc_avg_p())
    avgr.append(sim.calc_avg_r())
    (x, y, z) = sim.positions()  # extract new particle positions
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax3.set_xlim([-1, 1])
    ax3.set_ylim([-1, 1])

    ax6.set_xlim3d([-3, 3])
    ax6.set_ylim3d([-3, 3])
    ax6.set_zlim3d([-3, 3])

    scat.set_offsets(np.transpose(np.vstack((x, y))))
    scat2.set_offsets(np.transpose(np.vstack((y, z))))
    scat3.set_offsets(np.transpose(np.vstack((z, x))))
    scat4.set_data(x, y)
    scat4.set_3d_properties(z)
    pplot.set_data(t, avgp)
    rplot.set_data(t, avgr)
    ax4.set_xlim([0, max(t)])
    ax4.set_ylim([0, max(avgp)])
    ax5.set_xlim([0, max(t)])
    ax5.set_ylim([0, max(avgr)])
    com = sim.calc_com()
    com1.set_data(com[0], com[1])
    com2.set_data(com[1], com[2])
    com3.set_data(com[2], com[0])

    U, V = np.meshgrid(X, Y)

    meanfield1.set_UVC(0.00001 / (U - com[0]), 0.000001 / (V - com[1]))
    meanfield2.set_UVC(0.00001 / (U - com[1]), 0.000001 / (V - com[2]))
    meanfield3.set_UVC(0.00001 / (U - com[2]), 0.000001 / (V - com[0]))
    return [scat, scat2, scat3, pplot, rplot, com1, com2, com3, meanfield1, meanfield2, meanfield3, scat4]


anim = animation.FuncAnimation(fig, animate, interval=0.1, frames=100, blit=True)

plt.show()
