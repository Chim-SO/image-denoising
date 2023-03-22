from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.python.keras.models import Sequential


def encoder_decoder(input_dim):
    model = Sequential()
    model.add(Input(shape=input_dim))

    # Encoder:
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2), padding="same"))
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2), padding="same"))

    # Decoder:
    model.add(Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same"))
    model.add(Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same"))
    model.add(Conv2D(1, (3, 3), activation="sigmoid", padding="same"))

    return model
