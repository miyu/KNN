const add = (a, b) => a + b
const sum = (list) => list.reduce(add, 0)
const zip = (a, b, fn) => a.map((x, i) => fn ? fn(x, b[i]) : [x, b[i]])
const range = (n) => Array.apply(null, {length: n}).map((_, i) => i)
const squared_norm2 = (a, b) => sum(zip(a, b, (x, y) => (x - y) ** 2))
const uniform = (low, high) => low + Math.random() * (high - low)
const generate = (n, cb) => range(n).map(cb)
const random_points = (n) => generate(n, i => [uniform(-10, 10), uniform(-10, 10)])
const ground_truth_threshold = (x) =>  5 * Math.sin(x * 2 * Math.PI / 10)
const compute_label_knn = (q, points, labels, n) => {
  const dist = i => squared_norm2(q, points[i])
  const indices = range(points.length).sort((i1, i2) => dist(i1) - dist(i2)).slice(0, n)
  return sum(indices.map(i => Number(labels[i]))) > n / 2
};

const compute_label_ground_truth = (p) => ground_truth_threshold(p[0]) < p[1]

const training_points = random_points(1000)
const training_labels = training_points.map(compute_label_ground_truth)

const test_points = random_points(500)
const test_classifications = test_points.map(p => compute_label_knn(p, training_points, training_labels, 5))
const test_ground_truth_labels = test_points.map(compute_label_ground_truth)

const correct = sum(zip(test_classifications, test_ground_truth_labels, (a, b) => Number(a == b)))
const percent_correct = correct / test_points.length * 100
console.log(percent_correct, "percent of the classifications are correct.")
