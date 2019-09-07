import tensorflow as tf
import tensorflow_datasets as tfds

from easydict import EasyDict
from datetime import datetime
from variational_autoencoder.autoencoder import CVAE


def ld_cifar():
    def format_example(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    BATCH_SIZE = 128
    train_split_weights = (9, 1)
    train_split = tfds.Split.TRAIN.subsplit(weighted=train_split_weights)
    test_split = tfds.Split.TEST.subsplit(weighted=(1,))
    (raw_train, raw_validation, raw_test), metadata = tfds.load('cifar10', split=list(train_split + test_split),
                                                                with_info=True,
                                                                as_supervised=True)

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
    return EasyDict(train=train_batches, validation=validation_batches, test=test_batches,
                    steps=steps_dict)


def main():
    data = ld_cifar()
    model = CVAE(latent_dim=2)
    model.fit(data.train, epochs=2, steps_per_epoch=data.steps.train_steps, validation_data=data.validation,
              validation_steps=data.steps.validation_steps)
    model.evaluate(data.test, steps=data.steps.test_steps)
    # Save weights
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "./variational_autoencoder_logs/fit/" + now
    checkpoint_path = log_dir + "/weights/variational_autoencoder_weights"
    model.save_weights(checkpoint_path, save_format='tf')


if __name__ == '__main__':
    main()
