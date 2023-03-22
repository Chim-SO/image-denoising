import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.models.densenn.model import encoder_decoder

if __name__ == '__main__':
    # read dataset:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(f"The training data shape: {x_train.shape}, its label shape: {y_train.shape}")
    print(f"The test data shape: {x_test.shape}, its label shape: {y_test.shape}")

    # Scale images to the [0, 1] range:
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Reshape:
    num_pixels = 28 * 28
    x_train = x_train.reshape(60000, num_pixels).astype('float32')
    x_test = x_test.reshape(10000, num_pixels).astype('float32')
    print(f"The training data shape: {x_train.shape}")
    print(f"The test data shape: {x_test.shape}")

    # Noise:
    noise_level = 0.5
    x_noisy_train = x_train + noise_level * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_noisy_test = x_test + noise_level * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    x_noisy_train = np.clip(x_noisy_train, 0., 1.)
    x_noisy_test = np.clip(x_noisy_test, 0., 1.)

    # Model:
    model = encoder_decoder(num_pixels)
    model.summary()

    # Compile :
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train:
    model.fit(x_noisy_train, x_train, validation_data=(x_noisy_test, x_test), epochs=15, batch_size=256)

    # Display:
    n_imgs = 5
    predictions = np.reshape(model.predict(x_noisy_test[:n_imgs]), (-1, 28, 28))
    x_noisy_test = np.reshape(x_noisy_test, (-1, 28, 28))
    for i in range(n_imgs):
        # Print input image:
        plt.subplot(2, n_imgs, i + 1)
        plt.axis('off')
        plt.imshow(x_noisy_test[i], cmap='gray')
        # print output image:
        plt.subplot(2, n_imgs, n_imgs + i + 1)
        plt.axis('off')
        plt.imshow(predictions[i], cmap='gray')
    plt.show()
