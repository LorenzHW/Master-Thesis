import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from easydict import EasyDict
import os

from variational_autoencoder.autoencoder import CVAE, compute_loss
from variational_autoencoder.mnist_plotter import MNISTPlotter

FLAGS = flags.FLAGS


def ld_mnist():
    """Load training and test data."""

    def convert_types(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    dataset, info = tfds.load('mnist', data_dir='gs://tfds-data/datasets', with_info=True,
                              as_supervised=True)
    mnist_train, mnist_test = dataset['train'], dataset['test']
    mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(128)
    mnist_test = mnist_test.map(convert_types).batch(128)
    return EasyDict(train=mnist_train, test=mnist_test)


def generate_and_save_images(model, epoch, test_input):
    predictions = model.sample(test_input)

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    if not os.path.isdir("./images"):
        os.mkdir("./images")
    plt.savefig('./images/image_at_epoch_{:04d}.png'.format(epoch))


def main(_):
    latent_dim = 2
    num_examples_to_generate = 16

    data = ld_mnist()
    model = CVAE(latent_dim)
    optimizer = tf.keras.optimizers.Adam(1e-4)

    train_loss = tf.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    # Random vector constant for generation (prediction) so it will be easier to see the improvement.
    random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])

    @tf.function
    def train_step(x):
        with tf.GradientTape() as tape:
            loss, _ = compute_loss(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

    if FLAGS.train_new:
        for epoch in range(1, FLAGS.nb_epochs + 1):
            # keras like display of progress
            progress_bar_train = tf.keras.utils.Progbar(60000)
            for i, (x, y) in enumerate(data.train):
                train_step(x)
                progress_bar_train.add(x.shape[0], values=[('loss', train_loss.result())])

            # After every epoch generate an image
            generate_and_save_images(model, epoch, random_vector_for_generation)
        model.save_weights('./variational_autoencoder/weights/vae_weights', save_format='tf')
    else:
        model.load_weights('./variational_autoencoder/weights/vae_weights')

    progress_bar_test = tf.keras.utils.Progbar(10000)

    latent_representation_values = []
    for (x, y) in data.test:
        loss, z = compute_loss(model, x)
        latent_representation_values.append(z)
        test_loss(loss)
        elbo = -test_loss.result()
        progress_bar_test.add(x.shape[0], values=[('loss', elbo)])

    plotter = MNISTPlotter()
    plotter.plot_2D_latent_representation(latent_representation_values, data.test)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', 30, 'Number of epochs.')
    # flags.DEFINE_float('eps', 0.3, 'Total epsilon for FGM and PGD attacks.')
    flags.DEFINE_bool('train_new', False,
                      'If true a new model is trained and weights are saved to /weights, else weights are loaded from '
                      '/weights. Additionally, images are generated after every epoch.')
    app.run(main)
