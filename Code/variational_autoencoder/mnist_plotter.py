import matplotlib.pyplot as plt


class MNISTPlotter:
    def __init__(self):
        self.fig, self.axes = plt.subplots(1, 1)
        self.number_to_color = {
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
        self.static_param_dict = {
            "marker": "o",
            "linestyle": "None",

        }

    def plot_2D_latent_representation(self, latent_representation_values, test_data):
        number_to_dimensions = self._prepare_2D_latent_representation(latent_representation_values, test_data)
        for number, dimensions in number_to_dimensions.items():
            param_dict = {
                "color": self.number_to_color[number],
                "label": number
            }
            self.plot_data(dimensions[0], dimensions[1], param_dict)

    def plot_data(self, x, y, param_dict):
        # Merge with static param dict
        param_dict.update(self.static_param_dict)
        self.axes.plot(x, y, **param_dict)

    def _prepare_2D_latent_representation(self, z, data_test):
        numbers_to_dimensions = {}

        for i, (_, y_batch) in enumerate(data_test):
            z_batch = z[i].numpy()
            for j, number in enumerate(y_batch):
                number = str(number.numpy())
                dimension = z_batch[j]
                if number not in numbers_to_dimensions:
                    numbers_to_dimensions[number] = [[dimension[0]], [dimension[1]]]
                else:
                    numbers_to_dimensions[number][0].append(dimension[0])
                    numbers_to_dimensions[number][1].append(dimension[1])

        return numbers_to_dimensions

    def show_plot(self):
        plt.title("Latent representation of each number")
        plt.xlabel("First dimension")
        plt.ylabel("Second dimension")
        plt.legend()
        plt.show()
