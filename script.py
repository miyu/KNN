import matplotlib.pyplot as plt
import numpy as np
import operator

num_points = 5

# draw a graph with x-range [-10,10], y-range [-10,10]
# with a sin(x * 2pi/10)*5
x = np.arange(-10, 11)
y = np.sin(x * 2 * np.pi / 10) * 5
plt.plot(x, y)

# create the training points and test points
train_points = np.random.randint(low = -10, high = 10, size = (num_points, 2))
test_points = np.random.randint(low = -10, high = 10, size = (num_points, 2))

x_points = []
y_points = []
for i in range(num_points):
    x_points.append(train_points[i][0])
    y_points.append(train_points[i][1])

# Insert 500 points randomly within [-10, 10],[-10,10]. Label them "above" or
# "below" depending on whether they're above or below the wave.
plt.scatter(x_points, y_points)
plt.show()

# classifies training points as above or below the sine wave
def classify(points):
    s = (num_points, 3)
    classified = np.zeros(s)
    for point in range(num_points):
        x = points[point][0]
        y = points[point][1]
        sine_y = np.sin(x * 2 * np.pi / 10) * 5
        if sine_y < y:
            classification = True
        else:
            classification = False
        classified[point] = np.append(points[point], classification)
    return classified

classified_train_points = classify(train_points)
sorted = sorted(classified_train_points, key = (operator.itemgetter(0)))



def knn_classifier(points):
