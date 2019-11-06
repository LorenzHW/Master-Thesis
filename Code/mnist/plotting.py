from math import pi, sin, cos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import keras
from keras.utils import plot_model
import collections


def main():
    loss_points_xs = [100]
    loss_points_ys = [2]
    max_loss_points = pd.DataFrame({
        "dim_1": loss_points_xs,
        "dim_2": loss_points_ys
    })

    param_dict_anchor_points = {
        "s": 200
    }
    max_loss_point = max_loss_points.iloc[0]
    latent_rep_fig, latent_rep_axes = plt.subplots(1, 1, figsize=(10, 10))
    scatter_plot = latent_rep_axes.scatter(max_loss_point["dim_1"], max_loss_point["dim_2"],
                                           **param_dict_anchor_points)

    res = []
    for i in range(300):
        t = i / 30 * pi
        x = (1 + 5 * t) * cos(t)
        y = (1 + 5 * t) * sin(t)

        generated_sample = max_loss_point[["dim_1", "dim_2"]]
        generated_sample.at["dim_1"] += x
        generated_sample.at["dim_2"] += y
        res.append(generated_sample)

    res = pd.DataFrame(res)
    scatter_plot = latent_rep_axes.scatter(res["dim_1"].to_numpy(), res["dim_2"].to_numpy())
    latent_rep_fig.show()

    if not os.path.isdir("./images"):
        os.mkdir("./images")
    latent_rep_fig.savefig('images/walk.png')


def plot_walk_points():
    loss_points_xs = [1, 3]
    loss_points_ys = [2, 4]

    param_dict_anchor_points = {
        "s": 300
    }

    latent_rep_fig, latent_rep_axes = plt.subplots(1, 1, figsize=(10, 10))
    scatter_plot = latent_rep_axes.scatter(loss_points_xs, loss_points_ys, **param_dict_anchor_points)

    df = pd.DataFrame({
        "dim_1": loss_points_xs,
        "dim_2": loss_points_ys
    })

    first_point, second_point = df.iloc[0], df.iloc[1]
    direction_vector = second_point[["dim_1", "dim_2"]] - first_point[["dim_1", "dim_2"]]
    length_direction_vector = np.linalg.norm(direction_vector)
    unit_vector = 1 / np.linalg.norm(direction_vector) * direction_vector

    step_size = length_direction_vector / 5
    generated_points = []
    for i in range(1, 5):
        generated_sample = first_point[["dim_1", "dim_2"]] + i * step_size * unit_vector
        generated_points.append(generated_sample)
    generated_points = pd.DataFrame(generated_points)
    generated_points_xs = generated_points["dim_1"].to_numpy()
    generated_points_ys = generated_points["dim_2"].to_numpy()

    param_dict_generated_points = {
        "s": 300
    }
    latent_rep_axes.scatter(generated_points_xs, generated_points_ys, **param_dict_generated_points)
    latent_rep_fig.show()

    if not os.path.isdir("./images"):
        os.mkdir("./images")
    latent_rep_fig.savefig('images/walk.png')


def plot_random_sampling():
    loss_points_xs = [1.0]
    loss_points_ys = [2.0]
    max_loss_points = pd.DataFrame({
        "dim_1": loss_points_xs,
        "dim_2": loss_points_ys
    })

    param_dict_anchor_points = {
        "s": 200
    }
    anchor_point = max_loss_points.iloc[0]
    latent_rep_fig, latent_rep_axes = plt.subplots(1, 1, figsize=(10, 10))

    specified_range = 1.0
    res = []
    for _ in range(10000):
        generated_sample = anchor_point[["dim_1", "dim_2"]]
        for j in range(1, 3):
            generated_sample.at["dim_" + str(j)] += np.random.uniform(-specified_range, specified_range)
        res.append(generated_sample)
    res = pd.DataFrame(res)

    scatter_plot = latent_rep_axes.scatter(anchor_point["dim_1"], anchor_point["dim_2"],
                                           **param_dict_anchor_points)
    scatter_plot = latent_rep_axes.scatter(res["dim_1"].to_numpy(), res["dim_2"].to_numpy())
    if not os.path.isdir("./images"):
        os.mkdir("./images")
    latent_rep_fig.savefig('images/walk.png')
    latent_rep_fig.show()


def plot_centroids():
    class_1 = [0.0, 0.0]
    class_2 = [3.0, 3.0]
    class_3 = [3.0, 0.0]

    df1 = pd.DataFrame({
        "dim_1": [class_1[0]],
        "dim_2": [class_1[1]]
    })

    df2 = pd.DataFrame({
        "dim_1": [class_2[0]],
        "dim_2": [class_2[1]]
    })

    df3 = pd.DataFrame({
        "dim_1": [class_3[0]],
        "dim_2": [class_3[1]]
    })

    specified_range = 0.3
    res1 = []
    for _ in range(50):
        generated_sample = df1.iloc[0].copy()
        for j in range(1, 3):
            generated_sample.at["dim_" + str(j)] += np.random.uniform(-specified_range, specified_range)
        res1.append(generated_sample)
    res1 = pd.DataFrame(res1)

    specified_range = 0.3
    res2 = []
    for _ in range(50):
        generated_sample = df2.iloc[0].copy()
        for j in range(1, 3):
            generated_sample.at["dim_" + str(j)] += np.random.uniform(-specified_range, specified_range)
        res2.append(generated_sample)
    res2 = pd.DataFrame(res2)

    specified_range = 0.3
    res3 = []
    for _ in range(50):
        generated_sample = df3.iloc[0].copy()
        for j in range(1, 3):
            generated_sample.at["dim_" + str(j)] += np.random.uniform(-specified_range, specified_range)
        res3.append(generated_sample)
    res3 = pd.DataFrame(res3)

    latent_rep_fig, latent_rep_axes = plt.subplots(1, 1, figsize=(10, 10))
    scatter_plot = latent_rep_axes.scatter(res1["dim_1"].to_numpy(), res1["dim_2"].to_numpy())
    scatter_plot = latent_rep_axes.scatter(res2["dim_1"].to_numpy(), res2["dim_2"].to_numpy())
    scatter_plot = latent_rep_axes.scatter(res3["dim_1"].to_numpy(), res3["dim_2"].to_numpy())
    scatter_plot = latent_rep_axes.scatter(res1["dim_1"].mean(), res1["dim_2"].mean(), **{"c": "r"})
    scatter_plot = latent_rep_axes.scatter(res2["dim_1"].mean(), res2["dim_2"].mean(), **{"c": "r"})
    scatter_plot = latent_rep_axes.scatter(res3["dim_1"].mean(), res3["dim_2"].mean(), **{"c": "r"})
    scatter_plot = latent_rep_axes.scatter(0.5, 1, **{"c": "black"})

    latent_rep_fig.show()
    if not os.path.isdir("./images"):
        os.mkdir("./images")
    latent_rep_fig.savefig('images/assign_label.png')


def target_nn_model_plot():
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28, 1)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    plot_model(model, to_file="images/target_neural_net.png", show_shapes=True, show_layer_names=False)


def cvae_model_plot():
    class CVAE(tf.keras.Model):
        def __init__(self, latent_dim, hparams, logdir_path):
            super(CVAE, self).__init__()
            self.latent_dim = latent_dim  # TODO: Put into hparams
            self.vae_meta_information = collections.defaultdict(list)
            self.use_triplet_loss_for_backpropagration = False

            train_log_dir = logdir_path + '/gradient_tape/' + '/train'
            validation_log_dir = logdir_path + '/gradient_tape/' + '/validation'
            self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            self.validation_summary_writer = tf.summary.create_file_writer(validation_log_dir)

            str_to_optimizer_obj = {
                "adam": tf.keras.optimizers.Adam(1e-4),
                "sgd": tf.keras.optimizers.SGD()
            }
            self.optimizer = str_to_optimizer_obj[hparams["optimizer"]]

            self.inference_net = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                    tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                    tf.keras.layers.Flatten(),
                    # No activation
                    tf.keras.layers.Dense(latent_dim + latent_dim),
                ]
            )

            if hparams["use_batch_norm"] == "True":
                self.inference_net = tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                        tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                        tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Flatten(),
                        # No activation
                        tf.keras.layers.Dense(latent_dim + latent_dim),
                    ]
                )

            self.generative_net = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                    tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                    tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                    tf.keras.layers.Conv2DTranspose(
                        filters=64,
                        kernel_size=3,
                        strides=(2, 2),
                        padding="SAME",
                        activation='relu'),
                    tf.keras.layers.Conv2DTranspose(
                        filters=32,
                        kernel_size=3,
                        strides=(2, 2),
                        padding="SAME",
                        activation='relu'),
                    # No activation
                    tf.keras.layers.Conv2DTranspose(
                        filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
                ]
            )

    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(50,)),
            keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
            keras.layers.Reshape(target_shape=(7, 7, 32)),
            keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation='relu'),
            keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation='relu'),
            # No activation
            keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
        ]
    )
    plot_model(model, to_file="images/cvae_inference_net.png", show_shapes=True, show_layer_names=False)


if __name__ == '__main__':
    cvae_model_plot()
