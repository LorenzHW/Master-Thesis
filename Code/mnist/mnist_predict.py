from typing import List

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import collections
import pandas as pd
import requests, zipfile, io

from easydict import EasyDict

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

    def _generate_and_save_images(self, test_input):
        predictions = self._sample(test_input)
        return predictions

    def _decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


def ld_mnist():
    def format_example(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    BATCH_SIZE = 128
    train_split_weights = (9, 1)
    train_split = tfds.Split.TRAIN.subsplit(weighted=train_split_weights)
    test_split = tfds.Split.TEST.subsplit(weighted=(1,))
    (raw_train, raw_validation, raw_test), metadata = tfds.load('mnist', split=list(train_split + test_split),
                                                                with_info=True, as_supervised=True)

    additional_data, labels, num_generated_examples = generate_additional_data()
    num_copys = 5
    num_generated_examples *= num_copys
    temp_additional_data = additional_data
    temp_labels = labels
    for i in range(num_copys):
        temp_additional_data = tf.concat([temp_additional_data, additional_data], axis=0)
        temp_labels = tf.concat([temp_labels, labels], axis=0)

    additional_data_set = tf.data.Dataset.from_tensor_slices((temp_additional_data, temp_labels))

    train_batches = raw_train.map(format_example)
    train_batches = train_batches.concatenate(additional_data_set)  # Merge data sets
    train_batches = train_batches.shuffle(1000).batch(BATCH_SIZE)

    validation_batches = raw_validation.map(format_example).batch(BATCH_SIZE)
    test_batches = raw_test.map(format_example).batch(BATCH_SIZE)

    num_train, num_validation = (
        metadata.splits['train'].num_examples * weight / 10
        for weight in train_split_weights
    )

    num_train += num_generated_examples

    num_test = metadata.splits['test'].num_examples

    train_steps = round(num_train) // BATCH_SIZE
    validation_steps = round(num_validation) // BATCH_SIZE
    test_steps = round(num_test) // BATCH_SIZE

    steps_dict = EasyDict(train_steps=train_steps, validation_steps=validation_steps, test_steps=test_steps)
    data = EasyDict(train=train_batches, validation=validation_batches, test=test_batches, steps=steps_dict)

    return data


def generate_additional_data():
    latent_rep_vals_we_want_to_generate, num_dimensions = get_latent_representation_of_images_we_want_to_generate()
    dims_to_tensors = []
    for dim in range(1, num_dimensions + 1):
        dims_to_tensors.append(tf.convert_to_tensor(latent_rep_vals_we_want_to_generate["dimension_" + str(dim)]))
    latent_rep_values = tf.stack(dims_to_tensors, axis=1)

    hparams = {
        "optimizer": "adam",
        "use_batch_norm": "False"
    }
    model = CVAE(latent_dim=50, hparams=hparams, logdir_path="abc")
    r = requests.get(
        "https://github.com/LorenzHW/Master-Thesis/blob/master/Code/mnist/v_autoencoder/logs/20191014-081058/run-0/weights/weights.zip?raw=true")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()
    model.load_weights("./weights")

    generated_images = model._generate_and_save_images(latent_rep_values)
    labels = tf.convert_to_tensor(latent_rep_vals_we_want_to_generate["labels"], dtype=tf.int64)
    num_generated_examples = len(generated_images)
    return generated_images, labels, num_generated_examples


def get_latent_representation_of_images_we_want_to_generate():
    def get_remote_data(url):
        import requests
        import shutil
        response = requests.get(url, stream=True)
        with open('temp.npy', 'wb') as fin:
            shutil.copyfileobj(response.raw, fin)
        remote_data = np.load("temp.npy", allow_pickle=True)
        return remote_data.item()

    url = "https://github.com/LorenzHW/Master-Thesis/blob/master/Code/mnist/v_autoencoder/logs/20191014-081058/run-0/meta_info/values.npy?raw=true"
    vae_meta_info = get_remote_data(url)
    meta_info_key = "train_epoch_100"
    y_vs_pred = vae_meta_info.get(meta_info_key + "_y_vs_pred")
    external_model_loss_raw = vae_meta_info.get(meta_info_key + "_external_model_loss_raw")
    z = vae_meta_info.get(meta_info_key + "_z")
    label = vae_meta_info.get(meta_info_key + "_label")

    data = prepare_pd_df(z, external_model_loss_raw, label, y_vs_pred)
    wrongly_classified = data[data["y_vs_pred"] == 0]
    test = wrongly_classified.copy()
    num_dimensions = len(z[0][0])

    correctly_classified = data[data["y_vs_pred"] == 1]
    test = correctly_classified.sample(len(wrongly_classified))

    return wrongly_classified, num_dimensions


def prepare_pd_df(latent_values_batched: List[List], losses_batched: List, labels_batched, y_vs_pred_batched):
    num_dimensions = len(latent_values_batched[0][0])

    dims_separated = collections.defaultdict(list)
    for latent_values_batch in latent_values_batched:
        for dimensions in latent_values_batch:
            for i in range(num_dimensions):
                dims_separated["dimension_" + str(i + 1)].append(dimensions[i])

    other = {
        "labels": np.concatenate(labels_batched),
        "losses": np.concatenate(losses_batched),
        "y_vs_pred": np.concatenate(y_vs_pred_batched)
    }
    merged = {**dims_separated, **other}
    data = pd.DataFrame(merged)
    return data


def main():
    ld_mnist()


main()
