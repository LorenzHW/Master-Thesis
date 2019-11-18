import os
import tensorflow as tf
import numpy as np
import collections
import pandas as pd
import requests, zipfile, io
import matplotlib.pyplot as plt
from math import pi, sin, cos

import tensorflow_datasets as tfds
from easydict import EasyDict
from pandas import DataFrame, Series
from typing import List, Dict
from datetime import datetime

print(tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())

"""The Fast Gradient Method attack."""


def fast_gradient_method(model_fn, x, eps, norm, clip_min=None, clip_max=None, y=None,
                         targeted=False, sanity_checks=False):
    """
    Tensorflow 2.0 implementation of the Fast Gradient Method.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError("Norm order must be either np.inf, 1, or 2.")

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        asserts.append(tf.math.greater_equal(x, clip_min))

    if clip_max is not None:
        asserts.append(tf.math.less_equal(x, clip_max))

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        y = tf.argmax(model_fn(x), 1)

    grad = compute_gradient(model_fn, x, y, targeted)

    optimal_perturbation = optimize_linear(grad, eps, norm)
    # Add perturbation to original example to obtain adversarial example
    adv_x = x + optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        # We don't currently support one-sided clipping
        assert clip_min is not None and clip_max is not None
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x


# Due to performance reasons, this function is wrapped inside of tf.function decorator.
# Not using the decorator here, or letting the user wrap the attack in tf.function is way
# slower on Tensorflow 2.0.0-alpha0.
@tf.function
def compute_gradient(model_fn, x, y, targeted):
    """
    Computes the gradient of the loss with respect to the input tensor.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor
    :param y: Tensor with true labels. If targeted is true, then provide the target label.
    :param targeted:  bool. Is the attack targeted or untargeted? Untargeted, the default, will
                      try to make the label incorrect. Targeted will instead try to move in the
                      direction of being more like y.
    :return: A tensor containing the gradient of the loss with respect to the input tensor.
    """
    loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits
    with tf.GradientTape() as g:
        g.watch(x)
        # Compute loss
        loss = loss_fn(labels=y, logits=model_fn(x))
        if targeted:  # attack is targeted, minimize loss of target label rather than maximize loss of correct label
            loss = -loss

    # Define gradient of loss wrt input
    grad = g.gradient(loss, x)
    return grad


def optimize_linear(grad, eps, norm=np.inf):
    """
    Solves for the optimal input to a linear function under a norm constraint.

    Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)

    :param grad: tf tensor containing a batch of gradients
    :param eps: float scalar specifying size of constraint region
    :param norm: int specifying order of norm
    :returns:
      tf tensor containing optimal perturbation
    """

    # Convert the iterator returned by `range` into a list.
    axis = list(range(1, len(grad.get_shape())))
    avoid_zero_div = 1e-12
    if norm == np.inf:
        # Take sign of gradient
        optimal_perturbation = tf.sign(grad)
        # The following line should not change the numerical results. It applies only because
        # `optimal_perturbation` is the output of a `sign` op, which has zero derivative anyway.
        # It should not be applied for the other norms, where the perturbation has a non-zero derivative.
        optimal_perturbation = tf.stop_gradient(optimal_perturbation)
    elif norm == 1:
        abs_grad = tf.abs(grad)
        sign = tf.sign(grad)
        max_abs_grad = tf.reduce_max(abs_grad, axis, keepdims=True)
        tied_for_max = tf.dtypes.cast(tf.equal(abs_grad, max_abs_grad), dtype=tf.float32)
        num_ties = tf.reduce_sum(tied_for_max, axis, keepdims=True)
        optimal_perturbation = sign * tied_for_max / num_ties
    elif norm == 2:
        square = tf.maximum(avoid_zero_div, tf.reduce_sum(tf.square(grad), axis, keepdims=True))
        optimal_perturbation = grad / tf.sqrt(square)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are currently implemented.")

    # Scale perturbation to be the solution for the norm=eps rather than norm=1 problem
    scaled_perturbation = tf.multiply(eps, optimal_perturbation)
    return scaled_perturbation


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
                if not os.path.isdir("./downloaded/images"):
                    os.mkdir("./downloaded/images")
                plt.savefig('./downloaded/images/{}_image_{}.png'.format(image_prefix, i), bbox_inches='tight',
                            pad_inches=0)

        return predictions

    def _decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


class MyMethod:
    def __init__(self, num_samples_to_generate, option):
        self.num_samples_to_generate = num_samples_to_generate
        self.option = option
        self.vae_url = "https://github.com/LorenzHW/Master-Thesis/blob/master/Code/mnist/v_autoencoder/logs/20191111-173200/run-0/"
        self.model = self._load_variational_autoencoder()
        self.data, self.num_dimensions = self._prepare_external_metadata()
        self.centroids = self._compute_centroids()

    def _load_variational_autoencoder(self):
        hparams = {
            "optimizer": "adam",
            "use_batch_norm": "False"
        }
        model = CVAE(latent_dim=50, hparams=hparams, logdir_path="logs")
        r = requests.get(self.vae_url + "weights/weights.zip?raw=true")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall("./downloaded")
        model.load_weights("./downloaded/weights")
        return model

    def _load_external_metadata(self):
        def get_remote_data(url):
            import requests
            import shutil
            response = requests.get(url, stream=True)
            with open('./downloaded/meta_info.npy', 'wb') as fin:
                shutil.copyfileobj(response.raw, fin)
            remote_data = np.load("./downloaded/meta_info.npy", allow_pickle=True)
            return remote_data.item()

        meta_info_url = self.vae_url + "meta_info/values.npy?raw=true"
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

    @staticmethod
    def _get_max_loss_points_per_label(incorrect_classified_points_per_label: List[DataFrame]) -> List[DataFrame]:
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

    @staticmethod
    def biggest_spread_on_entire_latent_rep(incorrect_points_per_label: List[DataFrame]) -> List[DataFrame]:
        res = []
        for df in incorrect_points_per_label:
            incorrect_points = df.to_numpy()
            idxs_of_max_points = [0, 0]
            curr_max_distance = float("-inf")
            for idx1, p1 in enumerate(incorrect_points):
                for idx2, p2 in enumerate(incorrect_points):
                    l2_distance = np.linalg.norm(p1 - p2)
                    if l2_distance > curr_max_distance:
                        curr_max_distance = l2_distance
                        idxs_of_max_points = [idx1, idx2]
            combined = pd.concat([df.iloc[[idxs_of_max_points[0]]], df.iloc[[idxs_of_max_points[1]]]])
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

    def _walk_archimedean_spiral_from_anchor_point(self, anchor_point: Series) -> DataFrame:
        dimension_columns = ["dimension_" + str(i) for i in range(1, self.num_dimensions + 1)]
        res = []

        for i in range(self.num_samples_to_generate):
            t = i / 300 * pi
            x = (1 + 5 * t) * cos(t)
            y = (1 + 5 * t) * sin(t)

            generated_sample = anchor_point[dimension_columns]

            generated_sample.at["dimension_1"] += x
            generated_sample.at["dimension_2"] += y
            closest_centroid = self._determine_closest_centroid_label_for_point(generated_sample)
            generated_sample["label_of_nearest_centroid"] = closest_centroid
            res.append(generated_sample)

        res = pd.DataFrame(res)
        return res

    def _walk_from_anchor_point_to_anchor_point(self, anchor_points: DataFrame) -> DataFrame:
        res = []
        dimension_columns = ["dimension_" + str(i) for i in range(1, self.num_dimensions + 1)]
        first_point, second_point = anchor_points.iloc[1], anchor_points.iloc[0]
        direction_vector = second_point[dimension_columns] - first_point[dimension_columns]
        length_direction_vector = np.linalg.norm(direction_vector)
        unit_vector = 1 / np.linalg.norm(direction_vector) * direction_vector

        step_size = length_direction_vector / self.num_samples_to_generate
        for i in range(self.num_samples_to_generate):
            generated_sample = first_point[dimension_columns] + i * step_size * unit_vector
            closest_centroid = self._determine_closest_centroid_label_for_point(generated_sample)
            generated_sample["label_of_nearest_centroid"] = closest_centroid
            res.append(generated_sample)
        res = pd.DataFrame(res)
        return res

    def _generate_random_points_in_specific_range_for_single_point(self, anchor_point: Series) -> DataFrame:
        res = []
        dimension_columns = ["dimension_" + str(i) for i in range(1, self.num_dimensions + 1)]
        specified_range = 0.5

        for _ in range(self.num_samples_to_generate):
            generated_sample = anchor_point[dimension_columns]
            for j in range(1, self.num_dimensions + 1):
                generated_sample.at["dimension_" + str(j)] += np.random.uniform(-specified_range, specified_range)
            closest_centroid = self._determine_closest_centroid_label_for_point(generated_sample)
            generated_sample["label_of_nearest_centroid"] = closest_centroid
            res.append(generated_sample)
        res = pd.DataFrame(res)
        return res

    def generate_data(self):
        incorrect_classified_per_label = self._get_incorrect_classified_per_label()
        # anchor_points_per_label = self.biggest_spread_from_max_loss_point(incorrect_classified_per_label)

        if self.option in ["highest_loss_walk", "archimedean_spiral_walk", "generate_in_range_for_single_point"]:
            anchor_points_per_label = self._get_max_loss_points_per_label(incorrect_classified_per_label)
        elif self.option in ["biggest_spread_walk"]:
            anchor_points_per_label = self.biggest_spread_on_entire_latent_rep(incorrect_classified_per_label)
        anchor_points_per_label = self._add_label_of_closest_centroid_for_points(anchor_points_per_label)

        xs, ys, generate_png = [], [], False
        for anchor_points in anchor_points_per_label:
            cur_label = anchor_points.iloc[0]["labels"]

            if self.option == "archimedean_spiral_walk":
                z_df = self._walk_archimedean_spiral_from_anchor_point(anchor_points.iloc[-1])
            elif self.option in ["highest_loss_walk", "biggest_spread_walk"]:
                z_df = self._walk_from_anchor_point_to_anchor_point(anchor_points)
            elif self.option == "generate_in_range_for_single_point":
                z_df = self._generate_random_points_in_specific_range_for_single_point(anchor_points.iloc[-1])
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


def ld_mnist(use_vae_to_generate_data=False, samples_to_generate_per_label=250, option="archimedean_spiral_walk"):
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

    train_batches = raw_train.map(format_example)
    num_generated_examples = 0
    if use_vae_to_generate_data:
        m = MyMethod(samples_to_generate_per_label, option)
        print("GENERATING ADDITIONAL {} SAMPLES".format(samples_to_generate_per_label * 10))
        additional_data, labels, num_generated_examples = m.generate_data()
        num_copys = 1
        num_generated_examples *= num_copys

        additional_data_set = tf.data.Dataset.from_tensor_slices((additional_data, labels))
        train_batches = train_batches.concatenate(additional_data_set)  # Merge data sets

    train_batches = train_batches.shuffle(1000).batch(BATCH_SIZE).repeat()
    validation_batches = raw_validation.map(format_example).batch(BATCH_SIZE).repeat()
    test_batches = raw_test.map(format_example).batch(BATCH_SIZE).repeat()

    num_train, num_validation = (
        metadata.splits['train'].num_examples * weight / 10
        for weight in train_split_weights
    )

    num_train += num_generated_examples
    print("NUM TRAIN: ")
    print(num_train)

    num_test = metadata.splits['test'].num_examples

    train_steps = round(num_train) // BATCH_SIZE
    validation_steps = round(num_validation) // BATCH_SIZE
    test_steps = round(num_test) // BATCH_SIZE

    steps_dict = EasyDict(train_steps=train_steps, validation_steps=validation_steps, test_steps=test_steps)
    data = EasyDict(train=train_batches, validation=validation_batches, test=test_batches, steps=steps_dict)

    return data


def evaluate_on_adversarial(model, data):
    print("Evaluating on adversarial:")
    test_acc_fgsm = tf.metrics.SparseCategoricalAccuracy()
    steps = 0
    for idx, (x, y) in enumerate(data.test):
        x_fgm = fast_gradient_method(model, x, 0.3, np.inf)
        y_pred_fgm = model(x_fgm)
        test_acc_fgsm(y, y_pred_fgm)
        if steps >= data.steps.test_steps:
            break
        steps += 1

    print('test acc on FGM adversarial examples (%): {:.3f}'.format(test_acc_fgsm.result() * 100))


def main():
    use_vae_to_generate_data = False
    data = ld_mnist(use_vae_to_generate_data)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    with_additional_data = "with-additional-data/"
    if not use_vae_to_generate_data:
        with_additional_data = "without-additional-data/"

    logdir = "logs/" + with_additional_data + now + "/gradient-tape"
    weights_path = "logs/" + with_additional_data + now + "/weights"

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    model_cb = tf.keras.callbacks.ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True)
    es_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(data.train, epochs=1, steps_per_epoch=data.steps.train_steps, validation_data=data.validation,
              validation_steps=data.steps.validation_steps, callbacks=[tensorboard_callback, model_cb, es_cb])

    # Load best weights and evaluate
    model.load_weights(weights_path + "/variables/variables")
    model.evaluate(data.test, steps=data.steps.test_steps)
    evaluate_on_adversarial(model, data)


if __name__ == '__main__':
    main()

    # Used labels: 1, 6, 3, 4, 5
