from collections import namedtuple

import matplotlib.pyplot as plt
from typing import List, Dict

from tensorflow_core.python.ops.gen_dataset_ops import BatchDataset

PointsToLists = namedtuple("PointsConverted", ["xs", "ys", "losses"])
Point = namedtuple("Point", ["x", "y", "loss", "label"])


class Data:
    def __init__(self, xs: List, ys: List, labels: List, losses: List):
        """
        :param xs: List of data points for the x-axis
        :param ys: The data points for the y-axis
        :param labels: The label from data set, ML system was trained on
        :param losses: The losses from a external neural network
        """
        self.points = self._init_points(xs, ys, labels, losses)

    def map_labels_to_points(self) -> Dict[str, List[Point]]:
        result = {
            "0": [point for idx, point in enumerate(self.points) if point.label == "0"],
            "1": [point for idx, point in enumerate(self.points) if point.label == "1"],
            "2": [point for idx, point in enumerate(self.points) if point.label == "2"],
            "3": [point for idx, point in enumerate(self.points) if point.label == "3"],
            "4": [point for idx, point in enumerate(self.points) if point.label == "4"],
            "5": [point for idx, point in enumerate(self.points) if point.label == "5"],
            "6": [point for idx, point in enumerate(self.points) if point.label == "6"],
            "7": [point for idx, point in enumerate(self.points) if point.label == "7"],
            "8": [point for idx, point in enumerate(self.points) if point.label == "8"],
            "9": [point for idx, point in enumerate(self.points) if point.label == "9"],
        }
        return result

    @staticmethod
    def convert_points_to_lists(points: List[Point]) -> PointsToLists:
        xs = [point.x for point in points]
        ys = [point.y for point in points]
        losses = [point.loss for point in points]
        result = PointsToLists(xs, ys, losses)
        return result

    @staticmethod
    def _init_points(xs, ys, labels, losses) -> List[Point]:
        result = []
        for idx, x in enumerate(xs):
            result.append(Point(x, ys[idx], losses[idx], labels[idx]))
        return result


class MNISTPlotter:
    def __init__(self, zs: List, losses: List, test_data: BatchDataset):
        self.fig, self.axes = plt.subplots(2, 1, figsize=(20, 20))
        self.data = self._init_data(zs, losses, test_data)

    def plot_latent_representation(self):
        number_to_color = {
            "0": "blue",
            "1": "green",
            "2": "red",
            "3": "cyan",
            "4": "magenta",
            "5": "yellow",
            "6": "black",
            "7": "#808080",  # Gray
            "8": "#7214ec",  # Violet
            "9": "#3399ff"  # light blue
        }

        number_to_points = self.data.map_labels_to_points()
        for number, points in number_to_points.items():
            lists = self.data.convert_points_to_lists(points)
            param_dict = {
                "color": number_to_color[number],
                "label": number,
                "marker": "o",
                "linestyle": "None",
            }
            self.axes[0].plot(lists.xs, lists.ys, **param_dict)

    def plot_net_losses(self):
        lists = self.data.convert_points_to_lists(sorted(self.data.points, key=lambda p: p.loss, reverse=True)[:2])
        param_dict = {
            "cmap": plt.get_cmap("spring"),
            "c": lists.losses,
            "marker": "o",
            "linestyle": "None",

        }
        scatter_plot = self.axes[1].scatter(lists.xs, lists.ys, **param_dict)
        self.fig.colorbar(scatter_plot, ax=self.axes[1])

    @staticmethod
    def _init_data(zs: List[List], losses: List, test_data: BatchDataset) -> Data:
        xs, ys, ls, labels = [], [], [], []
        for i, (_, y_batch) in enumerate(test_data):
            for j, number in enumerate(y_batch):
                labels.append(str(number.numpy()))
                xs.append(zs[i][j][0].numpy())
                ys.append(zs[i][j][1].numpy())
                ls.append(losses[i][j].numpy())

        data = Data(xs=xs, ys=ys, labels=labels, losses=ls)
        return data

    def show_figure(self):
        self.axes[0].title.set_text("Latent representation of each number")
        self.axes[0].xaxis.label.set_text("First dimension")
        self.axes[0].yaxis.label.set_text("Second dimension")
        self.axes[0].legend()

        self.axes[1].title.set_text("TODO")

        self.fig.show()
