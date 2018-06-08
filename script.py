import matplotlib.pyplot as plt
import numpy as np

num_points = 10

# draw a graph with x-range [-10,10], y-range [-10,10]
# with a sin(x * 2pi/10)*5
x = np.arange(-10, 11)
y = np.sin(x * 2 * np.pi / 10) * 5
plt.plot(x, y)

# create the training points and test points
train_points = np.random.randint(low = -10, high = 10, size = (2, num_points))
test_points = np.random.randint(low = -10, high = 10, size = (2, num_points))

# Insert 500 points randomly within [-10, 10],[-10,10]. Label them "above" or
# "below" depending on whether they're above or below the wave.
plt.scatter(*train_points)
plt.show()

# classifies training points as above or below the sine wave
def classify(points):
    classifications = []
    classification = ""
    for point in range(num_points):
        x = points[0][point]
        y = points[1][point]
        print(x, y)
        sine_y = np.sin(x * 2 * np.pi / 10) * 5
        if sine_y < y:
            classification = "above"
        else:
            classification = "below"
        classifications.append(classification)
    return classifications

classified_train_points = classify(train_points)