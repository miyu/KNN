import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import random as r
import math

num_points = 9
k = 5

# draw a graph with x-range [-10,10], y-range [-10,10]
# with a sin(x * 2pi/10)*5
x = [i for i in range(-10, 11)]
y = [math.sin(i * 2.0 * math.pi / 10.0) * 5.0 for i in x]
plt.plot(x, y)

# create the training points and test points
train_points = [[r.randrange(start = -10, stop = 11), r.randrange(start = -10, stop = 11)]
                for i in range(num_points)]
test_points = [[r.randrange(start = -10, stop = 11), r.randrange(start = -10, stop = 11)]
                for i in range(num_points)]

# for plotting the points using pyplot
x_points = [x[0] for x in train_points]
y_points = [y[1] for y in train_points]

# Insert 500 points randomly within [-10, 10],[-10,10]. Label them "above" or
# "below" depending on whether they're above or below the wave.
plt.scatter(x_points, y_points)
plt.show()

# classifies training points as above (True) or below (False) the sine wave
def classify(points):
    classified = [[x, y, math.sin(x * 2 * math.pi / 10) * 5 < y] for [x, y] in points]
    return classified

classified_train_points = classify(train_points)
classified_test_points = classify(test_points)

# returns k indices of the training point distances closest to the given test point
def knn_classifier(k, train_points, test_point):
    test_x = test_point[0]
    test_y = test_point[1]
    distances = [math.sqrt(math.pow(test_x - train_x1, 2) + math.pow(test_y - train_y1, 2))
                 for train_x1, train_y1 in train_points]
    # print("Distances:", distances)

    # find k indices with the shortest distance to the test point
    temp = distances
    indexed_distances = [[i, temp[i]] for i in range(len(temp))]
    indexed_distances.sort(key=itemgetter(1))
    k_indices = [indexed_distances[index][0] for index in range(k)]

    # print("Indices of shortest distances of training points to the test point:", k_indices)
    return k_indices

# classifies each test point
tested_classifications = []
for test_point in test_points:
    indices = knn_classifier(k, train_points, test_point)
    values = [classified_train_points[index][2] for index in indices]

    # count the majority classifications (above or below)
    print(values)
    num_train_points_above = sum(values)
    num_train_points_below = k - num_train_points_above
    print(num_train_points_above, num_train_points_below)
    # print("Training Points Above:", num_train_points_above)

    classification = num_train_points_above > num_train_points_below

    tested_classifications.append(classification)

    # print("The test point", test_point, "is predicted to be", str_classification, "the sine wave.")

# calculate percent of calculations that are correct
correct = 0
for m in range(len(classified_test_points)):
    if classified_test_points[m][2] == tested_classifications[m]:
        correct = correct + 1

percent_correct = correct / len(tested_classifications) * 100
print(percent_correct, "percent of the classifications are correct.")
