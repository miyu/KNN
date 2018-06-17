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


def knn_classifier(k, train_points, test_point):
    distances = []
    test_x = test_point[0]
    test_y = test_point[1]
    for point in range(num_points):
        train_x = train_points[point][0]
        train_y = train_points[point][1]
        distance = np.sqrt(np.square(test_x - train_x) + np.square(test_y - train_y))
        distances = np.append(distances, distance)
    print(distances)

    a = (num_points, 2)
    indexed_distances = np.array(a)
    for i in range(len(distances)):
        indexed_distances.append([])
        indexed_distances[i].append(i)
        indexed_distances[i].append(distances[i])

    print(indexed_distances)
    distances_sorted = sorted(indexed_distances, key=(operator.itemgetter(1)))
    print(distances_sorted)



for i in range(num_points):
    test_point = test_points[i]
    knn_classifier(3, train_points, test_point)
