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

def plot_labeled_points(points, labels, is_test):
    for ((x, y), label) in list(zip(points, labels)):
        marker = 'x' if is_test else '.'
        color = 'r' if label else 'b'
        plt.plot([x], [y], marker + color)

def ground_truth_threshold(x):
    return 5 * sin(x * 2 * pi / 10)

def compute_label_knn(q, training_data, n=5, distance_function=squared_norm2):
    knn = heapq.nsmallest(n, training_data, key=lambda item: distance_function(q, item[0]))
    return sum(int(label) for (p, label) in knn) > n / 2

def compute_label_ground_truth(p):
    return ground_truth_threshold(p[0]) < p[1]

# plot ground truth
plot_points([(x, ground_truth_threshold(x)) for x in arange(-10, 10, 0.01)])

# plot training data
training_points = random_points(1000)
training_labels = [compute_label_ground_truth(p) for p in training_points]
training_data = list(zip(training_points, training_labels))
plot_labeled_points(training_points, training_labels, False)

# plot test data
test_points = random_points(500)
test_classifications = [compute_label_knn(p, training_data) for p in test_points]
test_ground_truth_labels = [compute_label_ground_truth(p) for p in test_points]
plot_labeled_points(test_points, test_classifications, True)
plt.show()

# calculate percent of calculations that are correct
correct = sum(a == b for (a, b) in zip(test_classifications, test_ground_truth_labels))
percent_correct = correct / len(test_points) * 100
print(percent_correct, "percent of the classifications are correct.")
