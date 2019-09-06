from easydict import EasyDict
import tensorflow_datasets as tfds
import tensorflow as tf

IMG_SIZE = 96
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
basemodel_path = "mobilenet"


def ld_cifar():
    def format_example(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    BATCH_SIZE = 128
    train_split_weights = (9, 1)
    train_split = tfds.Split.TRAIN.subsplit(weighted=train_split_weights)
    test_split = tfds.Split.TEST.subsplit(weighted=(1,))
    (raw_train, raw_validation, raw_test), metadata = tfds.load('cifar10', split=list(train_split + test_split),
                                                                with_info=True,
                                                                as_supervised=True)

    train_batches = raw_train.map(format_example).shuffle(1000).batch(BATCH_SIZE).repeat()
    validation_batches = raw_validation.map(format_example).batch(BATCH_SIZE).repeat()
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
    feature_extractor_layer = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet',
                                                                input_shape=IMG_SHAPE)
    feature_extractor_layer.trainable = False

    model = tf.keras.Sequential([
        feature_extractor_layer,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.summary()

    opt = tf.keras.optimizers.SGD()
    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['acc'])

    log_dir = "./" + basemodel_path + "_logs/fit/20190906-074935"
    checkpoint_path = log_dir + "/weights/" + basemodel_path + "_weights"

    model.load_weights(checkpoint_path)  # Load best model
    result = model.evaluate(data.test, steps=data.steps.test_steps)
    print("Loss on test data: " + str(result[0]))
    print("Accuracy on test: data" + str(result[1]))
    print(result)


if __name__ == "__main__":
    main()
