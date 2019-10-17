import os
from typing import List

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import collections
import pandas as pd
import requests, zipfile, io
import matplotlib.pyplot as plt
from easydict import EasyDict
from sklearn.cluster import DBSCAN

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

    def generate_and_save_images(self, test_input, generate_images=False, image_prefix=""):
        predictions = self._sample(test_input)

        if generate_images:
            for i in range(predictions.shape[0]):
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
    def __init__(self):
        self.model = self._load_variational_autoencoder()
        self.data, self.num_dimensions = self._prepare_external_metadata()
        pass

    def _load_variational_autoencoder(self):
        hparams = {
            "optimizer": "adam",
            "use_batch_norm": "False"
        }
        model = CVAE(latent_dim=50, hparams=hparams, logdir_path="abc")
        r = requests.get(
            "https://github.com/LorenzHW/Master-Thesis/blob/master/Code/mnist/v_autoencoder/logs/20191017-080757/run-0/weights/weights.zip?raw=true")
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

        meta_info_url = "https://github.com/LorenzHW/Master-Thesis/blob/master/Code/mnist/v_autoencoder/logs/20191017-080757/run-0/meta_info/values.npy?raw=true"
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

    def _construct_clusters(self):
        incorrect_classified_samples = self.data[self.data["y_vs_pred"] == False]
        dimension_columns = ["dimension_" + str(i) for i in range(1, self.num_dimensions + 1)]
        labels = incorrect_classified_samples["labels"].unique()
        clusters = []

        for l in labels:
            l_incorrect_classified = incorrect_classified_samples[incorrect_classified_samples["labels"] == l]
            db = DBSCAN(eps=10, min_samples=5).fit(l_incorrect_classified[dimension_columns])
            l_incorrect_classified["clustered"] = db.labels_
            main_cluster = l_incorrect_classified[l_incorrect_classified["clustered"] == 0]
            clusters.append(main_cluster)
        return clusters

    def _convert_dims_to_tensors(self, dataframe):
        tensors = []
        for dim in range(1, self.num_dimensions + 1):
            tensors.append(tf.convert_to_tensor(dataframe["dimension_" + str(dim)]))
        return tf.stack(tensors, axis=1)

    def generate_data(self):
        clusters = self._construct_clusters()
        maxed_clusters = self.max_based_on_external_clf(clusters)
        zs = []
        for c in maxed_clusters:
            zs.append(self._convert_dims_to_tensors(c))

        xs, ys = [], []
        for i, z in enumerate(zs):
            curr_label = str(clusters[i]["labels"].unique()[0])

            images = self.model.generate_and_save_images(z, True, "cluser_with_label_" + curr_label)
            labels = tf.convert_to_tensor(clusters[i]["labels"], dtype=tf.int64)
            # for j, img in enumerate(images): # Flatten
            #     xs.append(img)
            #     ys.append(labels[j])
            print("HELLO WORLD")

        xs = np.array(xs).flatten()
        ys = np.array(ys).flatten()
        num_generated_examples = len(xs)
        return xs, ys, num_generated_examples

    def max_based_on_external_clf(self, clusters):
        maxed_clusters = []
        for c in clusters:
            c = c.sort_values(by=["losses"])
            maxed_clusters.append(c.tail(3))

        return maxed_clusters


def main():
    m = MyMethod()
    m.generate_data()


if __name__ == '__main__':
    main()
