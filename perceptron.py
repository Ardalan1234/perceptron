# Ardalan Omidrad May 21, 2023 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, points: dict):
        self.p = np.array([key for key, value in points.items()])
        self.t = np.array([value for key, value in points.items()])
        self.shape = len(self.p[0])
        self.w = np.random.randn(self.shape)
        self.b = np.random.randn(1)
        self.plot_range = np.min(self.p) - 1, np.max(self.p) + 1
        self.previous_w_b = []

    def hardlim(self, x):
        return 1 if x >= 0 else 0

    def update(self, w_old, b_old, p, error):
        self.w = w_old + error * p
        self.b = b_old + error

    def dividing_line(self, i_range: tuple, last=False):

        x = np.linspace(*self.plot_range, 10)

        for i in range(*i_range):
            # plotting w
            """plt.arrow(0, 0, self.previous_w_b[i][0][0], self.previous_w_b[i][0][-1],
                      head_width=0.07, head_length=0.1, fc='k')"""

            p2_intercept = -self.previous_w_b[i][-1] / self.previous_w_b[i][0][-1]
            p1_intercept = -self.previous_w_b[i][-1] / self.previous_w_b[i][0][0]

            gradient = (p2_intercept - 0) / (0 - p1_intercept)
            c = -(gradient * p1_intercept)
            y = gradient * x + c

            plt.plot(x, y, c='c' if not last else 'm', zorder=-1, alpha=0.001 if not last else 1)

    def plot(self):
        x1 = [i[0] for i in self.p]
        y1 = [j[1] for j in self.p]
        plt.scatter(x1, y1, s=50, c=['r' if t == 0 else 'k' for t in self.t], alpha=1)

        self.dividing_line(i_range=(0, len(self.previous_w_b)))

        plt.xlim(self.plot_range)
        plt.ylim(self.plot_range)

        self.dividing_line(i_range=(-1, -2, -1), last=True)
        plt.show()

    def predict(self, point):
        predictions = []
        for p in point:
            res = self.hardlim(self.w.dot(p.T) + self.b)
            predictions.append(res)
        return predictions

    def accuracy(self, predictions, lbl):
        return np.sum(predictions == lbl) / len(lbl)

    def main(self):
        run = True

        while run:
            for i in range(len(self.p)):
                a = self.hardlim(self.w.dot(self.p[i].T) + self.b)
                e = self.t[i] - a
                if e == 0:
                    continue
                else:
                    self.previous_w_b.append((self.w, self.b))
                    self.update(self.w, self.b, self.p[i], e)
                    break
            else:
                self.previous_w_b.append((self.w, self.b))
                run = False
        self.plot()
        return self.w, self.b


# model = Perceptron({(1, 2): 1, (-1, 2): 0, (0, -1): 0})
# model = Perceptron({(0, 0): 1, (0, 1): 1, (-1, 0): 1, (-2, 0): 0, (-1, 1): 0, (0, 2): 0})
# model = Perceptron({(-2, 2): 0, (0, 2): 0, (2, 2): 0, (-2, 0): 0, (2, 0): 0, (-2, -2): 1, (0, -2): 1, (2, -2): 1})
# model = Perceptron({(-2, 2): 0, (0, 2): 1, (2, 2): 1, (-2, 0): 1, (2, 0): 1, (-2, -2): 1, (0, -2): 1, (2, -2): 1})
# model = Perceptron({(-2, 2): 1, (0, 2): 1, (2, 2): 0, (-2, 0): 1, (2, 0): 0, (-2, -2): 1, (0, -2): 0, (2, -2): 0})
"""
model = Perceptron(
    {(-1, -1): 1, (-1, -2): 1, (-1, -3): 1, (0, -1): 1, (0, -4): 1, (0, -3): 1, (-3, -3): 1, (-3, 0): 1, (-2, -1): 1,
     (-2, 1): 1, (-2, -2): 1, (-1, 0): 1, (1, -1): 1, (2, 1): 0, (5, -1): 0, (3, 1): 0, (3, 2): 0, (3, 3): 0, (4, 1): 0,
     (1, 2): 0, (2, -2): 1, (4, 5): 0, (1, 0): 1, (0, 1): 1, (-1, 2): 1, (0, 2): 0, (1, 3): 0, (2, 3): 0,
     (2, 4): 0, (4, 0): 0, (4, 2): 0, (4, 4): 0, (3, 4): 0, (3, -1): 1}
)
"""
# load data
data = pd.read_csv("data.csv")
data = np.array(data, dtype=np.int)

np.random.shuffle(data)

points = data[:, 0:-1]
labels = data[:, -1]

# setup data
x_train, x_test = np.split(points, [8, 10])[-1], np.split(points, [8, 10])[0]
y_train, y_test = np.split(labels, [8, 10])[-1], np.split(labels, [8, 10])[0]

model = Perceptron({tuple(d[0]): d[-1] for d in zip(x_train, y_train)})
w, b = model.main()
model_predictions = model.predict(x_test)
print(f"w: {w}\nb: {b}")
print(f"model prediction: {model_predictions}")
print(f"real label:       {list(y_test)}")
print(f"accuracy:         {model.accuracy(model_predictions, y_test) * 100} %")
