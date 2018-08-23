import matplotlib.pyplot as plt
import numpy as np
import operator
import random as r
import math

num_points = 9

# draw a graph with x-range [-10,10], y-range [-10,10]
# with a sin(x * 2pi/10)*5
x = [i for i in range(-10, 11)]
y = [math.sin(i * 2.0 * math.pi / 10.0) * 5.0 for i in x]
plt.plot(x, y)

# create the training points and test points
train_points = [[r.randrange(start = -10, stop = 10), r.randrange(start = -10, stop = 10)]
                for i in range(num_points)]
# train_points = np.random.randint(low = -10, high = 10, size = (num_points, 2))
test_points = np.random.randint(low = -10, high = 10, size = (num_points, 2))

# for plotting the points using pyplot
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
# sorted = sorted(classified_train_points, key = (operator.itemgetter(0)))

# returns k indices of the training point distances closest to the given test point
def knn_classifier(k, train_points, test_point):
    distances = []
    test_x = test_point[0]
    test_y = test_point[1]
    for point in range(num_points):
        train_x = train_points[point][0]
        train_y = train_points[point][1]
        distance = np.sqrt(np.square(test_x - train_x) + np.square(test_y - train_y))
        distances = np.append(distances, distance)
    # print("Distances:", distances)

    # find k indices with the shortest distance to the test point
    indices = []
    t = 0
    while len(indices) < k:
        shortest = distances[t]
        index = t
        while shortest == -1.0:
            shortest = distances[t + 1]
            index = t + 1
        for i in range(1, len(distances)):
            if distances[i] != -1.0 and distances[i] < shortest:
                shortest = distances[i]
                index = i
        indices.append(index)
        distances[index] = -1

    # print("Indices of shortest distances of training points to the test point:", indices)
    return indices

# classifies each test point
tested_classifications = []
for i in range(num_points):
    test_point = test_points[i]
    indices = knn_classifier(3, train_points, test_point)
    values = []
    for index in indices:
        values.append(classified_train_points[index][2])

    # count the majority classifications (above or below)
    above_count = 0
    below_count = 0
    for j in values:
        if j == 1.0:
            above_count = above_count + 1
        else:
            below_count = below_count + 1
    # print("Training Points Above:", above_count, "| Training Points Below:", below_count)

    str_classification = "unknown"

    if above_count > below_count:
        str_classification = "above"
        classification = 1
    else:
        str_classification = "below"
        classification = 0

    tested_classifications.append(classification)

    # print("The test point", test_point, "is predicted to be", str_classification, "the sine wave.")
    # print()

# calculate percent of calculations that are correct
classified_test_points = classify(test_points)
correct = 0
for m in range(len(classified_test_points)):
    if classified_test_points[m][2] == tested_classifications[m]:
        correct = correct + 1

percent_correct = correct / len(tested_classifications) * 100
print(percent_correct, "percent of the classifications are correct.")
