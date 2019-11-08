import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os
import collections
import requests, zipfile, io

from tensorboard.plugins.hparams import api as hp
from easydict import EasyDict
from datetime import datetime
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

print(tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(distances, 0.0), dtype=tf.float32)
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the maximum.

    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
    masked_maximums = math_ops.reduce_max(
        math_ops.multiply(data - axis_minimums, mask), dim,
        keepdims=True) + axis_minimums
    return masked_maximums


def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the minimum.

    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
    masked_minimums = math_ops.reduce_min(
        math_ops.multiply(data - axis_maximums, mask), dim,
        keepdims=True) + axis_maximums
    return masked_minimums


def triplet_loss_adapted_from_tf(y_true, y_pred):
    margin = 1.
    labels = y_true
    embeddings = y_pred

    ### Code from Tensorflow function [tf.contrib.losses.metric_learning.triplet_semihard_loss] starts here:

    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = array_ops.shape(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    pdist_matrix = _pairwise_distances(embeddings, squared=True)
    # Build pairwise binary adjacency matrix.
    adjacency = math_ops.equal(labels, array_ops.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = math_ops.logical_not(adjacency)

    # global batch_size
    batch_size = array_ops.size(labels)  # was 'array_ops.size(labels)'

    # Compute the mask.
    pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
    mask = math_ops.logical_and(
        array_ops.tile(adjacency_not, [batch_size, 1]),
        math_ops.greater(
            pdist_matrix_tile, array_ops.reshape(
                array_ops.transpose(pdist_matrix), [-1, 1])))
    mask_final = array_ops.reshape(
        math_ops.greater(
            math_ops.reduce_sum(
                math_ops.cast(mask, dtype=tf.float32), 1, keepdims=True),
            0.0), [batch_size, batch_size])
    mask_final = array_ops.transpose(mask_final)

    adjacency_not = math_ops.cast(adjacency_not, dtype=tf.float32)
    mask = math_ops.cast(mask, dtype=tf.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = array_ops.reshape(
        masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = array_ops.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = array_ops.tile(
        masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
    semi_hard_negatives = array_ops.where(
        mask_final, negatives_outside, negatives_inside)

    loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = math_ops.cast(
        adjacency, dtype=tf.float32) - array_ops.diag(
        array_ops.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = math_ops.reduce_sum(mask_positives)

    semi_hard_triplet_loss_distance = math_ops.truediv(
        math_ops.reduce_sum(
            math_ops.maximum(
                math_ops.multiply(loss_mat, mask_positives), 0.0)),
        num_positives,
        name='triplet_semihard_loss')

    ### Code from Tensorflow function semi-hard triplet loss ENDS here.
    return semi_hard_triplet_loss_distance


external_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

external_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
r = requests.get(
    "https://github.com/LorenzHW/Master-Thesis/blob/master/Code/mnist/classifier/without-additional-data/20191014-164731/weights/variables/variables.zip?raw=true")
zipf = zipfile.ZipFile(io.BytesIO(r.content))
zipf.extractall()
external_model.load_weights("./variables")


# Convolutional Variational Autoencoder
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

    @tf.function
    def _get_external_model_info(self, x_batch, y):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        predictions = external_model(x_batch)
        loss_raw = loss_object(y, predictions)
        y_vs_pred = tf.math.equal(y, tf.argmax(predictions, axis=1))
        loss_binary = tf.map_fn(lambda loss: 1 if loss > 0.005 else 0, loss_raw, dtype=tf.int32)

        return loss_raw, loss_binary, y_vs_pred

    def _sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self._decode(eps, apply_sigmoid=True)

    def _generate_and_save_images(model, epoch, test_input):
        predictions = model._sample(test_input)

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.axis('off')

        if not os.path.isdir("./images"):
            os.mkdir("./images")
        plt.savefig('./images/image_at_epoch_{:04d}.png'.format(epoch))

    def _encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    @staticmethod
    def _reparameterize(mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def _log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)

        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def compute_loss(self, x_image):
        """ Computes the evidence lower bound (ELBO) which should get maximized
        """
        mean, logvar = self._encode(x_image)
        z = self._reparameterize(mean, logvar)
        x_logit = self._decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x_image)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self._log_normal_pdf(z, 0., 0.)
        logqz_x = self._log_normal_pdf(z, mean, logvar)
        loss_mean = -tf.reduce_mean(logpx_z + logpz - logqz_x)
        loss_per_sample = -(logpx_z + logpz - logqz_x)
        return loss_mean, loss_per_sample

    def compute_triplet_loss(self, x_image, y):
        mean, logvar = self._encode(x_image)
        z = self._reparameterize(mean, logvar)
        triplet_loss = triplet_loss_adapted_from_tf(y, z)
        return triplet_loss, z

    def _decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    @tf.function
    def _train_step(self, x_image, y_for_triplet_loss):
        with tf.GradientTape() as tape:
            elbo_loss, elbo_loss_per_sample = self.compute_loss(x_image)
            triplet_loss, z = self.compute_triplet_loss(x_image, y_for_triplet_loss)
            loss = elbo_loss
            if self.use_triplet_loss_for_backpropagration:
                loss = triplet_loss + elbo_loss

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return triplet_loss, elbo_loss, z, elbo_loss_per_sample

    @tf.function
    def _validate_step(self, x_image, y_for_triplet_loss):
        elbo_loss, elbo_loss_per_sample = self.compute_loss(x_image)
        triplet_loss, z = self.compute_triplet_loss(x_image, y_for_triplet_loss)
        return triplet_loss, elbo_loss, z, elbo_loss_per_sample

    def _train(self, epoch, data):
        # keras like display of progress
        triplet_loss_metric, elbo_loss_metric = tf.metrics.Mean(name='triplet_loss'), tf.metrics.Mean(name='elbo')
        meta_info_key = "train_epoch_{}".format(epoch)

        progress_bar_train = tf.keras.utils.Progbar(data.steps.train_steps + 1)
        for batch_idx, (x_image, y) in enumerate(data.train):
            external_model_loss_raw, y_for_triplet_loss, y_vs_pred = self._get_external_model_info(x_image, y)

            triplet_loss, elbo_loss, z, elbo_loss_per_sample = self._train_step(x_image, y_for_triplet_loss)
            triplet_loss_metric(triplet_loss)
            elbo_loss_metric(elbo_loss)
            if epoch % 100 == 0:
                self._add_meta_info(meta_info_key, z, y, y_vs_pred, y_for_triplet_loss, external_model_loss_raw,
                                    elbo_loss_per_sample)
            progress_bar_train.add(1, values=[('triplet_loss', triplet_loss_metric.result()),
                                              ('elbo_loss', elbo_loss_metric.result())])
        return triplet_loss_metric.result(), elbo_loss_metric.result()

    def _validate(self, epoch, data):
        elbo_loss_metric, triplet_loss_metric = tf.metrics.Mean(name='validation_loss_elbo'), tf.metrics.Mean(
            name='validation_loss_triplet')
        meta_info_key_validation = "validation_epoch_{}".format(epoch)

        for batch_idx, (x_image, y) in enumerate(data.validation):
            external_model_loss_raw, y_for_triplet_loss, y_vs_pred = self._get_external_model_info(x_image, y)

            triplet_loss, elbo_loss, z, elbo_loss_per_sample = self._validate_step(x_image, y_for_triplet_loss)
            triplet_loss_metric(triplet_loss)
            elbo_loss_metric(elbo_loss)
            if epoch % 100 == 0:
                self._add_meta_info(meta_info_key_validation, z, y, y_vs_pred, y_for_triplet_loss,
                                    external_model_loss_raw, elbo_loss_per_sample)
        print("Validation loss triplet: {:.4f}; Validation loss elbo {:.4f}".format(triplet_loss_metric.result(),
                                                                                    elbo_loss_metric.result()))
        return triplet_loss_metric.result(), elbo_loss_metric.result()

    def _add_meta_info(self, meta_info_key, z, y, y_vs_pred, external_model_loss_binary, external_model_loss_raw,
                       elbo_per_sample):
        self.vae_meta_information[meta_info_key + "_y_vs_pred"].append(y_vs_pred)
        self.vae_meta_information[meta_info_key + "_external_model_loss_binary"].append(external_model_loss_binary)
        self.vae_meta_information[meta_info_key + "_external_model_loss_raw"].append(external_model_loss_raw)
        self.vae_meta_information[meta_info_key + "_z"].append(z.numpy())
        self.vae_meta_information[meta_info_key + "_label"].append(y.numpy())
        self.vae_meta_information[meta_info_key + "_elbo"].append(elbo_per_sample.numpy())

    def fit(self, epochs, data):
        for epoch in range(1, epochs + 1):

            info_string = "TRAINING CVAE WITH TRIPLET LOSS AND ELBO"
            if not self.use_triplet_loss_for_backpropagration:
                info_string = 'TRAINING CVAE ONLY WITH ELBO'
            print("Epoch {}/{} - {}".format(epoch, epochs, info_string))

            triplet_loss, elbo_loss = self._train(epoch, data)
            with self.train_summary_writer.as_default():
                tf.summary.scalar('triplet_loss', triplet_loss, step=epoch)
                tf.summary.scalar('elbo', elbo_loss, step=epoch)

            triplet_loss, elbo_loss = self._validate(epoch, data)
            with self.validation_summary_writer.as_default():
                tf.summary.scalar('triplet_loss', triplet_loss, step=epoch)
                tf.summary.scalar('elbo', elbo_loss, step=epoch)

    def evaluate(self, data):
        print("Evaluating:")
        progress_bar_test = tf.keras.utils.Progbar(data.steps.test_steps + 1)
        elbo_loss_metric = tf.metrics.Mean(name='test_loss_elbo')
        triplet_loss_metric = tf.metrics.Mean(name='test_loss_triplet')
        for batch_idx, (x, y) in enumerate(data.test):
            external_model_loss_raw, y_for_triplet_loss, y_vs_pred = self._get_external_model_info(x, y)

            triplet_loss, elbo_loss, z, elbo_loss_per_sample = self._validate_step(x, y_for_triplet_loss)
            elbo_loss_metric(elbo_loss)
            triplet_loss_metric(triplet_loss)
            self._add_meta_info("test_data", z, y, y_vs_pred, y_for_triplet_loss, external_model_loss_raw,
                                elbo_loss_per_sample)
            progress_bar_test.add(1, values=[('triplet_loss', triplet_loss_metric.result()),
                                             ('elbo_loss', elbo_loss_metric.result())])
        return elbo_loss_metric.result(), triplet_loss_metric.result()

    def save_meta_info(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        np.save(path + "/values.npy", self.vae_meta_information)


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

    train_batches = raw_train.map(format_example).shuffle(1000).batch(BATCH_SIZE)
    validation_batches = raw_validation.map(format_example).batch(BATCH_SIZE)
    test_batches = raw_test.map(format_example).batch(BATCH_SIZE)

    num_train, num_validation = (
        metadata.splits['train'].num_examples * weight / 10
        for weight in train_split_weights
    )

    num_test = metadata.splits['test'].num_examples

    train_steps = round(num_train) // BATCH_SIZE
    validation_steps = round(num_validation) // BATCH_SIZE
    test_steps = round(num_test) // BATCH_SIZE

    steps_dict = EasyDict(train_steps=train_steps, validation_steps=validation_steps, test_steps=test_steps)
    data = EasyDict(train=train_batches, validation=validation_batches, test=test_batches, steps=steps_dict)

    return data


def run(run_dir, hparams, data):
    with tf.summary.create_file_writer(run_dir + "/hparam_tuning").as_default():
        load_weights = True
        model = CVAE(latent_dim=50, hparams=hparams, logdir_path=run_dir)
        if load_weights:
            r = requests.get(
                "https://github.com/LorenzHW/Master-Thesis/blob/master/Code/mnist/v_autoencoder/logs/20191108-094553/run-0/weights/weights.zip?raw=true")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall()
            model.load_weights("./weights")

        model.fit(epochs=100, data=data)
        elbo_loss, triplet_loss = model.evaluate(data)

        checkpoint_path = run_dir + "/weights/weights"
        meta_info_path = run_dir + "/meta_info"
        model.save_weights(checkpoint_path, save_format='tf')
        model.save_meta_info(meta_info_path)

        # Record data for TensorBoard
        hp.hparams(hparams)  # record the values used in this trial
        tf.summary.scalar("elbo", elbo_loss, step=1)
        tf.summary.scalar("triplet_loss", triplet_loss, step=1)


def main():
    session_num, data = 0, ld_mnist()
    hp_optimizers = ["adam"]
    hp_use_batch_normalization = ["False"]
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    for use_batch_norm in hp_use_batch_normalization:
        for optimizer in hp_optimizers:
            hparams = {
                "optimizer": optimizer,
                "use_batch_norm": use_batch_norm,
            }

            run_name = "run-{}".format(session_num)
            print('--- Starting trial: {}'.format(run_name))
            run('logs/' + current_time + "/" + run_name, hparams, data)
            session_num += 1


if __name__ == '__main__':
    main()
