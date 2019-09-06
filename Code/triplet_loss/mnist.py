import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from easydict import EasyDict
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import datetime

from triplet_loss.triplet_loss import batch_hard_triplet_loss

FLAGS = flags.FLAGS


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


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


def main(_):
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(y, predictions)

    # @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)

        for idx, p in enumerate(predictions):
            tf.argmax(p)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    data = ld_mnist()
    model = MyModel()
    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    train_loss = tf.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    if FLAGS.train_new:
        for epoch in range(1, FLAGS.nb_epochs + 1):
            # keras like display of progress
            progress_bar_train = tf.keras.utils.Progbar(60000)
            for i, (x, y) in enumerate(data.train):
                train_step(x, y)
                progress_bar_train.add(x.shape[0], values=[('loss', train_loss.result())])
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        model.save_weights('./triplet_loss/weights/weights', save_format='tf')
    else:
        model.load_weights('./triplet_loss/weights/weights')

    progress_bar_test = tf.keras.utils.Progbar(10000)
    for (x, y) in data.test:
        test_step(x, y)
        progress_bar_test.add(x.shape[0], values=[('accuracy', test_accuracy.result())])


if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', 5, 'Number of epochs.')
    # flags.DEFINE_float('eps', 0.3, 'Total epsilon for FGM and PGD attacks.')
    flags.DEFINE_bool('train_new', False,
                      'If true a new model is trained and weights are saved to /weights, else weights are loaded from '
                      '/weights. Additionally, images are generated after every epoch.')
    app.run(main)
