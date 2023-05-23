# Ardalan Omidrad
# May 21, 2023
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

    @staticmethod
    def hardlim(x):
        return 1 if x >= 0 else 0

    def __update(self, w_old, b_old, p, error):
        self.w = w_old + error * p
        self.b = b_old + error

    def __dividing_line(self, i_range: tuple, last=False):

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

            plt.plot(x, y, c='c' if not last else 'm', zorder=-1, alpha=0.01 if not last else 1)

    def __plot(self):
        x1 = [i[0] for i in self.p]
        y1 = [j[1] for j in self.p]
        plt.scatter(x1, y1, s=50, c=['r' if t == 0 else 'k' for t in self.t], alpha=1)

        self.__dividing_line(i_range=(0, len(self.previous_w_b)))

        plt.xlim(self.plot_range)
        plt.ylim(self.plot_range)

        self.__dividing_line(i_range=(-1, -2, -1), last=True)
        plt.show()

    def predict(self, point):
        predictions = []
        for p in point:
            res = self.hardlim(self.w.dot(p.T) + self.b)
            predictions.append(res)
        return predictions

    @staticmethod
    def accuracy(predictions, lbl):
        return np.sum(predictions == lbl) / len(lbl)

    def single_prediction(self, point: [list, tuple]):
        return self.hardlim(self.w.dot(np.array(point).T) + self.b)

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
                    self.__update(self.w, self.b, self.p[i], e)
                    break
            else:
                self.previous_w_b.append((self.w, self.b))
                run = False

        self.__plot()
        return self.w, self.b


# load data
data = pd.read_csv("data.csv")
data = np.array(data)

np.random.shuffle(data)

points = data[:, 0:-1]
labels = data[:, -1]

# setup data
x_train, x_test = np.split(points, [int(0.8 * len(data))])
y_train, y_test = np.split(labels, [int(0.8 * len(data))])

model = Perceptron({tuple(d[0]): d[-1] for d in zip(x_train, y_train)})
w, b = model.main()
model_predictions = model.predict(x_test)
print(f"w: {w}\nb: {b}")
print(f"model prediction: {model_predictions}")
print(f"real label:       {list(y_test)}")
print(f"accuracy:         {Perceptron.accuracy(model_predictions, y_test) * 100} %")
print(f"prediction:       {model.single_prediction([120, 45])}")
