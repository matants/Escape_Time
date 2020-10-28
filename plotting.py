import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

dp = 0.01
p1 = np.arange(0, 1, dp)
p2 = np.arange(0, 1, dp)
P1, P2 = np.meshgrid(p1, p2)


def escape_time_expect_2d(P1, P2):
    return 2 + 1 + 2 * (P1 / (1 - P1) + P2 / (1 - P2) + P1 * P2 / ((1 - P1) * (1 - P2)))


T = escape_time_expect_2d(P1, P2)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(P1, P2, T, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()
