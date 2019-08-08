import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from easydict import EasyDict

FLAGS = flags.FLAGS

IMG_SIZE = 32
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


def ld_cifar():
    def format_example(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    BATCH_SIZE = 32
    SPLIT_WEIGHTS = (8, 2)
    splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)

    (raw_train, raw_test), metadata = tfds.load('cifar10', split=list(splits), with_info=True, as_supervised=True)

    train_batches = raw_train.map(format_example).shuffle(1000).batch(BATCH_SIZE).repeat()
    test_batches = raw_test.map(format_example).batch(BATCH_SIZE)

    num_train, num_test = (
        metadata.splits['train'].num_examples * weight / 10
        for weight in SPLIT_WEIGHTS
    )

    steps_per_epoch = round(num_train) // BATCH_SIZE
    return EasyDict(train=train_batches, test=test_batches, steps_per_epoch=steps_per_epoch)


def main(_):
    data = ld_cifar()
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=IMG_SHAPE),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(32, (3, 3)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0, 25),

        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, (3, 3)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Activation('softmax')
    ])

    opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(data.train, epochs=FLAGS.nb_epochs, steps_per_epoch=data.steps_per_epoch)

    if FLAGS.train_new:
        model.fit(data.train, epochs=FLAGS.nb_epochs, steps_per_epoch=data.steps_per_epoch)
        model.save_weights("./cifar-10/weights/own_weights", save_format="tf")
    else:
        model.load_weights("./cifar-10/weights/own_weights")


if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs.')
    flags.DEFINE_bool('train_new', True,
                      'If true a new model is trained and weights are saved to /weights, else weights are loaded from '
                      '/weights. Additionally, images are generated after every epoch.')

    app.run(main)
