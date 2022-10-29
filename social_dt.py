# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# %%
w = 0.5  # <1
c1 = 2
c2 = 2
p = 10
n = 30
r1 = np.random.uniform(0.3, 0.7, n)
r2 = np.random.uniform(0.3, 0.7, n)

# %%
Vx = [0]*n
Vy = [0]*n
global VxMax
VxMax = 0

personal_best_arr = []
global_best = (1, 0)

particles_arr = [(-10+20*np.random.randint(1, 30), 0) for i in range(n)]

personal_best_arr = particles_arr.copy()

# %%


def function(x, y):
    return 160-15*pow(math.sin(2*x), 2)-(x-2)**2


# %%
def update_personal_best(particles_arr, personal_best_arr):
    for particle in particles_arr:
        new_val = function(particle[0], particle[1])

        if new_val > function(personal_best_arr[particles_arr.index(particle)][0], personal_best_arr[particles_arr.index(particle)][1]):
            personal_best_arr[particles_arr.index(particle)] = particle

    return personal_best_arr

# %%


def update_global_best(personal_best_arr, global_best):
    for particle in personal_best_arr:
        new_val = function(particle[0], particle[1])

        if new_val > function(global_best[0], global_best[1]):
            global_best = particle

    return global_best

# %%


def update_velocity(Vx, Vy, particles_arr, personal_best_arr, global_best):
    for i in range(n):
        Vx[i] = w*Vx[i]+c1*r1[i]*(personal_best_arr[i][0]-particles_arr[i]
                                  [0])+c2*r2[i]*(global_best[0]-particles_arr[i][0])
        Vy[i] = w*Vy[i]+c1*r1[i]*(personal_best_arr[i][1]-particles_arr[i]
                                  [1])+c2*r2[i]*(global_best[1]-particles_arr[i][1])

        global VxMax

        VxMax = max(VxMax, Vx[i])
        if(Vx[i] > 20):
            Vx[i] = 20
    return Vx, Vy

# %%


def update_cordinated(particles_arr, Vx, Vy):
    for i in range(n):
        particles_arr[i] = (particles_arr[i][0]+Vx[i],
                            particles_arr[i][1]+Vy[i])
    return particles_arr


# %%
if __name__ == "__main__":

    plt.show()

    for i in range(100):

        personal_best_arr = update_personal_best(
            particles_arr, personal_best_arr)
        global_best = update_global_best(personal_best_arr, global_best)
        Vx, Vy = update_velocity(
            Vx, Vy, particles_arr, personal_best_arr, global_best)
        particles_arr = update_cordinated(particles_arr, Vx, Vy)

        particles_arr_x = [x for (x, _) in particles_arr]
        particles_arr_y = [y for (_, y) in particles_arr]

        plt.xlim(-2,2)
        plt.ylim(-2,2)

        plt.plot(particles_arr_x, particles_arr_y, 'bo')

        plt.draw()
        plt.pause(0.01)
        plt.clf()
    print(global_best)

    print(function(global_best[0], 0))

    # print(VxMax)

    plt.show()
