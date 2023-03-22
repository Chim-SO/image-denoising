
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.models.cnn.model import encoder_decoder

if __name__ == '__main__':
    # read dataset:
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    print(f"The training data shape: {x_train.shape}")
    print(f"The test data shape: {x_test.shape}")

    # Scale images to the [0, 1] range:
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Input dimension transformation:
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Noise:
    noise_level = 0.5
    x_noisy_train = x_train + noise_level * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_noisy_test = x_test + noise_level * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    x_noisy_train = np.clip(x_noisy_train, 0., 1.)
    x_noisy_test = np.clip(x_noisy_test, 0., 1.)

    # Encoder decoder
    model = encoder_decoder((28, 28, 1))
    model.compile(loss="binary_crossentropy", optimizer="adam")

    # Train:
    model.fit(x_noisy_train, x_train, validation_data=(x_noisy_test, x_test), epochs=5, batch_size=200)

    # Display:
    n_imgs = 5
    predictions = model.predict(x_noisy_test[:n_imgs])*255
    for i in range(n_imgs):
        # Print input image:
        plt.subplot(2, n_imgs, i + 1)
        plt.axis('off')
        plt.imshow(x_noisy_test[i], cmap='gray')
        # print output image:
        plt.subplot(2, n_imgs, n_imgs +i + 1)
        plt.axis('off')
        plt.imshow(predictions[i], cmap='gray')
    plt.show()
