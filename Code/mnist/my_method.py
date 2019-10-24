import os
import tensorflow as tf
import numpy as np
import collections
import pandas as pd
import requests, zipfile, io
import matplotlib.pyplot as plt

from pandas import DataFrame, Series
from typing import List

print(tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())


# Convolutional Variational Autoencoder
class CVAE(tf.keras.Model):
    def __init__(self, latent_dim, hparams, logdir_path):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim  # TODO: Put into hparams
        self.vae_meta_information = collections.defaultdict(list)
        self.use_triplet_loss_for_backpropagration = True

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

    def _sample(self, z=None):
        if z is None:
            z = tf.random.normal(shape=(100, self.latent_dim))
        return self._decode(z, apply_sigmoid=True)

    def generate_and_save_images(self, test_input, generate_images=False, label=None, generated_imgs=None):
        predictions = self._sample(test_input)

        if generate_images:
            for i in range(predictions.shape[0]):
                nearest_centroid_label = generated_imgs.iloc[i]["label_of_nearest_centroid"]
                image_prefix = "max_loss_point_label_{}_nearest_centroid_label_{}".format(label, nearest_centroid_label)

                plt.subplot(1, 1, 1)
                plt.imshow(predictions[i, :, :, 0], cmap='gray')
                plt.axis('off')
                if not os.path.isdir("./images"):
                    os.mkdir("./images")
                plt.savefig('images/{}_image_{}.png'.format(image_prefix, i))

        return predictions

    def _decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


class MyMethod:
    def __init__(self, num_samples_to_generate):
        self.model = self._load_variational_autoencoder()
        self.data, self.num_dimensions = self._prepare_external_metadata()
        self.centroids = self._compute_centroids()
        self.num_samples_to_generate = num_samples_to_generate
        pass

    def _load_variational_autoencoder(self):
        hparams = {
            "optimizer": "adam",
            "use_batch_norm": "False"
        }
        model = CVAE(latent_dim=50, hparams=hparams, logdir_path="abc")
        r = requests.get(
            "https://github.com/LorenzHW/Master-Thesis/blob/master/Code/fashion_mnist/v_autoencoder/logs/20191024-094817/run-0/weights/weights.zip?raw=true")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
        model.load_weights("./weights")
        return model

    def _load_external_metadata(self):
        def get_remote_data(url):
            import requests
            import shutil
            response = requests.get(url, stream=True)
            with open('meta_info.npy', 'wb') as fin:
                shutil.copyfileobj(response.raw, fin)
            remote_data = np.load("meta_info.npy", allow_pickle=True)
            return remote_data.item()

        meta_info_url = "https://github.com/LorenzHW/Master-Thesis/blob/master/Code/fashion_mnist/v_autoencoder/logs/20191024-094817/run-0/meta_info/values.npy?raw=true"
        vae_meta_info, meta_info_key = get_remote_data(meta_info_url), "train_epoch_100"
        y_vs_pred_batched = vae_meta_info.get(meta_info_key + "_y_vs_pred")
        external_model_loss_raw_batched = vae_meta_info.get(meta_info_key + "_external_model_loss_raw")
        z_batched, labels_batched = vae_meta_info.get(meta_info_key + "_z"), vae_meta_info.get(meta_info_key + "_label")
        elbo_batched = vae_meta_info.get(meta_info_key + "_elbo")
        return z_batched, labels_batched, external_model_loss_raw_batched, y_vs_pred_batched, elbo_batched

    def _prepare_external_metadata(self):
        latent_values_batched, labels_batched, losses_batched, y_vs_pred_batched, elbo_batched = self._load_external_metadata()

        num_dimensions = len(latent_values_batched[0][0])

        dims_separated = collections.defaultdict(list)
        for latent_values_batch in latent_values_batched:
            for dimensions in latent_values_batch:
                for i in range(num_dimensions):
                    dims_separated["dimension_" + str(i + 1)].append(dimensions[i])

        other = {
            "labels": np.concatenate(labels_batched),
            "losses": np.concatenate(losses_batched),
            "y_vs_pred": np.concatenate(y_vs_pred_batched),
            "elbo": np.concatenate(elbo_batched)
        }
        merged = {**dims_separated, **other}
        data = pd.DataFrame(merged)
        return data, num_dimensions

    def _get_incorrect_classified_per_label(self) -> List[DataFrame]:
        incorrect_classified_samples = self.data[self.data["y_vs_pred"] == False]
        labels = incorrect_classified_samples["labels"].unique()
        res = []

        for l in labels:
            l_incorrect_classified = incorrect_classified_samples[incorrect_classified_samples["labels"] == l]
            res.append(l_incorrect_classified)
        return res

    def _convert_dims_to_tensors(self, dataframe):
        tensors = []
        for dim in range(1, self.num_dimensions + 1):
            tensors.append(tf.convert_to_tensor(dataframe["dimension_" + str(dim)]))
        return tf.stack(tensors, axis=1)

    def _get_max_loss_points_per_label(self, incorrect_classified_points_per_label: List[DataFrame]) -> List[DataFrame]:
        max_loss_dfs = []
        for df in incorrect_classified_points_per_label:
            df = df.sort_values(by=["losses"])
            max_loss_dfs.append(df.tail(2))

        return max_loss_dfs

    def biggest_spread_from_max_loss_point(self, incorrect_points_per_label: List[DataFrame]) -> List[DataFrame]:
        dimension_columns = ["dimension_" + str(i) for i in range(1, self.num_dimensions + 1)]
        res = []

        for df in incorrect_points_per_label:
            df = df.sort_values(by=["losses"])
            max_loss_point = df.tail(1)

            cur_max_distance = float("-inf")
            cur_max_idx = 0
            points = df[dimension_columns].to_numpy()
            for idx, p in enumerate(points):
                l2_distance = np.linalg.norm(max_loss_point[dimension_columns].to_numpy()[0] - p)
                if l2_distance > cur_max_distance:
                    cur_max_distance = l2_distance
                    cur_max_idx = idx
            furthest_away = df.iloc[[cur_max_idx]]
            combined = pd.concat([max_loss_point, furthest_away])
            res.append(combined)
        return res

    def _compute_centroids(self) -> List[DataFrame]:
        labels = self.data["labels"].unique()
        centroids_for_labels = []

        for l in labels:
            centroid = pd.DataFrame({
                "label": [l]
            })
            data_for_label = self.data[self.data["labels"] == l]
            for n in range(1, self.num_dimensions + 1):
                mean_of_dimension = data_for_label["dimension_" + str(n)].mean()
                centroid["dimension_" + str(n)] = mean_of_dimension
            centroids_for_labels.append(centroid)
        return centroids_for_labels

    def _add_label_of_closest_centroid_for_points(self, max_loss_points_per_label: List[DataFrame]):

        for points in max_loss_points_per_label:
            points["label_of_nearest_centroid"] = None  # Create additional column
            column_idx = points.columns.get_loc("label_of_nearest_centroid")
            points.iat[0, column_idx] = self._determine_closest_centroid_label_for_point(points.iloc[0])
            points.iat[1, column_idx] = self._determine_closest_centroid_label_for_point(points.iloc[1])

        return max_loss_points_per_label

    def _determine_closest_centroid_label_for_point(self, p: Series) -> int:
        dimension_columns = ["dimension_" + str(i) for i in range(1, self.num_dimensions + 1)]
        curr_min_distance, curr_label = float("inf"), None
        for idx, c in enumerate(self.centroids):
            l2_distance = np.linalg.norm(c[dimension_columns].to_numpy()[0] - p[dimension_columns].to_numpy())
            if l2_distance < curr_min_distance:
                curr_min_distance = l2_distance
                curr_label = c["label"][0]
        return curr_label

    def _walk_from_first_point_to_second_point(self, points: DataFrame) -> DataFrame:
        res = []
        dimension_columns = ["dimension_" + str(i) for i in range(1, self.num_dimensions + 1)]
        first_point, second_point = points.iloc[0], points.iloc[1]
        direction_vector = second_point[dimension_columns] - first_point[dimension_columns]
        length_direction_vector = np.linalg.norm(direction_vector)
        unit_vector = 1 / np.linalg.norm(direction_vector) * direction_vector

        step_size = length_direction_vector / self.num_samples_to_generate
        for i in range(self.num_samples_to_generate):
            generated_sample = first_point[dimension_columns] + i * step_size * unit_vector
            closest_centroid = self._determine_closest_centroid_label_for_point(generated_sample)
            generated_sample["label_of_nearest_centroid"] = closest_centroid
            res.append(generated_sample)
        res.append(second_point[dimension_columns + ["label_of_nearest_centroid"]])
        res = pd.DataFrame(res)
        return res

    def generate_data(self):
        incorrect_classified_per_label = self._get_incorrect_classified_per_label()
        max_loss_points_per_label = self.biggest_spread_from_max_loss_point(incorrect_classified_per_label)
        # max_loss_points_per_label = self._get_max_loss_points_per_label(incorrect_classified_per_label)
        max_loss_points_per_label = self._add_label_of_closest_centroid_for_points(max_loss_points_per_label)

        xs, ys, generate_png = [], [], False
        for cur_max_loss_p in max_loss_points_per_label:
            cur_label = cur_max_loss_p.iloc[0]["labels"]
            z_df = self._walk_from_first_point_to_second_point(cur_max_loss_p)
            z = self._convert_dims_to_tensors(z_df)
            if generate_png:
                images = self.model.generate_and_save_images(z, True, cur_label, z_df)
            else:
                images = self.model.generate_and_save_images(z)
            labels = tf.convert_to_tensor(
                [int(z_df.iloc[i]["label_of_nearest_centroid"]) for i in range(z_df.shape[0])], dtype=tf.int64
            )
            for j, img in enumerate(images):  # Flatten
                xs.append(img)
                ys.append(labels[j])

        num_generated_examples = len(xs)
        return xs, ys, num_generated_examples


def main():
    m = MyMethod(10)
    m.generate_data()


if __name__ == '__main__':
    main()
