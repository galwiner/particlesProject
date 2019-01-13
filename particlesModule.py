import numpy as np


class particle:
    def __init__(self, r=np.zeros(3), p=np.zeros(3), ownField=lambda x: np.zeros(3), m=1):
        self.r = r
        self.p = p
        self.m = 1

    #         self.ownField=lambda r: np.array([0,0,0])-self.r
    #         self.m=arc.Rubidium87.mass
    def ownField(self,r):
        try:
            return 0*1 / ((r - self.r))
        finally:
            return 0*np.random.uniform(-5,5,(1,3))

    def __repr__(self):
        return ("particle in r=[{0},{1},{2}],p=[{3},{4},{5}]".format(*self.r, *self.p))

def gaussianBeam(k,wavelength=1):
    '''returns the force field due to a gaussian beam with k wavevector and lambda '''

    pass

class simulator:
    def __init__(self, particles=[], ownField=np.zeros((1, 3)), forceFunc=lambda r, p: np.array([0, 0, 0]), dt=0.01,
                 Tend=0.01):
        self.particles = particles
        self.Tend = Tend
        self.dt = dt
        self.forceFunc = forceFunc
        # self.ownField = ownField

    #         self.meanField=self.calcMeanField()

    def positions(self):
        '''return positions of all particles'''
        N = np.size(self.particles)
        space = np.concatenate([particle.r for particle in self.particles]).reshape(N, 3)
        return ((space[:, 0], space[:, 1], space[:, 2]))

    #     def calcMeanField(self):
    #         def sigma(funcs, r):
    #             return sum(f(r) for f in funcs)
    #         def meanField(r):
    #             return sigma([p.ownField for p in self.particles],r)
    #         return meanField
    def calc_avg_p(self):
        return np.mean([np.linalg.norm(particle.p) for particle in self.particles])

    def calc_avg_r(self):
        return np.mean([np.linalg.norm(particle.r) for particle in self.particles])

    def calc_com(self):
        return np.mean(self.positions(), 1)

    def mean_field(self, r):
        return np.sum([particle.ownField(r) for particle in self.particles])*(np.linalg.norm(r)<0.5)

    def project(self):
        '''project the particle space onto the XZ plane'''

        (x, y, z) = space = self.positions()
        return (x, z)

    def step(self):
        for particle in self.particles:
            particle.p = particle.p + self.dt * (self.forceFunc(particle.r, particle.p)+self.mean_field(particle.r))
            particle.r = particle.r + self.dt * particle.p / particle.m

    def run(self):
        T = 0
        while (T < self.Tend):
            T = T + self.dt
            self.step()

    def __repr__(self):
        return ("simulator with {} particles".format(np.size(self.particles)))


def animator(simulator):
    pass
