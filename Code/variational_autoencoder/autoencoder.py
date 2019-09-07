import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os


# Convolutional Variational Autoencoder
class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim

        self.inference_net = tf.keras.Sequential(
            [

                tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=8 * 8 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(8, 8, 32)),
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
                    filters=3, kernel_size=3, strides=(1, 1), padding="SAME"),
            ]
        )

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

    def compute_loss(self, x):
        mean, logvar = self._encode(x)
        z = self._reparameterize(mean, logvar)
        x_logit = self._decode(z)
        # TODO: for evaluation save z values somewhere
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self._log_normal_pdf(z, 0., 0.)
        logqz_x = self._log_normal_pdf(z, mean, logvar)

        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def _decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits

    def fit(self, train_data, epochs, steps_per_epoch, validation_data, validation_steps):
        @tf.function
        def train_step(x):
            with tf.GradientTape() as tape:
                loss = self.compute_loss(x)
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            return loss

        optimizer = tf.keras.optimizers.Adam(1e-4)
        for epoch in range(1, epochs + 1):
            # keras like display of progress
            train_loss = tf.metrics.Mean(name='train_loss')
            progress_bar_train = tf.keras.utils.Progbar(steps_per_epoch + 1)
            print("Epoch {}/{}".format(epoch, epochs))
            for x, y in train_data:
                loss = train_step(x)
                train_loss(loss)
                progress_bar_train.add(1, values=[('loss', train_loss.result())])

            validation_loss = tf.metrics.Mean(name='validation_loss')
            for (x, y) in validation_data:
                loss = self.compute_loss(x)
                validation_loss(loss)
            print("Validation loss: {}".format(validation_loss.result()))

        # TODO: Generate picture after each epoch?
        # TODO: How would transfer learning affect the autoencoder?

    def evaluate(self, test_data, steps):
        @tf.function
        def test_step(x):
            loss = self.compute_loss(x)
            return loss

        progress_bar_test = tf.keras.utils.Progbar(steps + 1)
        test_loss = tf.metrics.Mean(name='test_loss')
        print("Evaluating:")
        for (x, y) in test_data:
            loss = test_step(x)
            test_loss(loss)
            progress_bar_test.add(1, values=[('loss', test_loss.result())])
