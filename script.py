import matplotlib.pyplot as plt
import numpy as np

# draw a graph with x-range [-10,10], y-range [-10,10]
# with a sin(x * 2pi/10)*5
x = np.arange(-10, 11)
y = np.sin(x * 2 * np.pi / 10) * 5
plt.plot(x, y)

# Insert 500 points randomly within [-10, 10],[-10,10]. Label them true or
# false depending on whether they're above or below the wave.
plt.scatter(*np.random.randint(low = -10, high = 10, size = (2, 500)))
plt.show()

# Build a KNN classifier with that & classify new points.