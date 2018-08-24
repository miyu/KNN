import matplotlib.pyplot as plt
from random import uniform
from math import sin, pi
from numpy import arange
import heapq

def squared_norm2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def random_points(n, low = -10, high = 10):
    return [(uniform(low, high), uniform(low, high)) for _ in range(n)]

def plot_points(points):
    plt.plot([p[0] for p in points], [p[1] for p in points])

def plot_labeled_points(labeled_points, is_test):
    for ((x, y), label) in labeled_points:
        marker = 'x' if is_test else '.'
        color = 'r' if label else 'b'
        plt.plot([x], [y], marker + color)

def ground_truth_threshold(x):
    return 5 * sin(x * 2 * pi / 10)

def label_points_ground_truth(points):
    labels = [ground_truth_threshold(x) < y for (x, y) in points]
    return list(zip(points, labels))

def label_point_knn(q, training_data, n=5, distance_function=squared_norm2):
    knn = heapq.nsmallest(n, training_data, key=lambda item: distance_function(q, item[0]))
    return (q, sum([int(label) for (p, label) in knn]) > n / 2)

def label_points_knn(qs, training_data, n=5, distance_function=squared_norm2):
    return [label_point_knn(q, training_data, n, distance_function) for q in qs]

# plot ground truth
plot_points([(x, ground_truth_threshold(x)) for x in arange(-10, 10, 0.01)])

# plot training data
labeled_training_points = label_points_ground_truth(random_points(100))
plot_labeled_points(labeled_training_points, False)

# plot test data
test_points = random_points(500)
classified_test_points = label_points_knn(test_points, labeled_training_points)
plot_labeled_points(classified_test_points, True)
plt.show()

# calculate percent of calculations that are correct
ground_truth_test_points = label_points_ground_truth(test_points)
correct = sum([a[1] == b[1] for (a, b) in zip(classified_test_points, ground_truth_test_points)])
percent_correct = correct / len(test_points) * 100
print(percent_correct, "percent of the classifications are correct.")
