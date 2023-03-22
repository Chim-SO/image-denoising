from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential


def encoder_decoder(input_dim):
    model = Sequential()
    # Encoder
    model.add(Dense(500, input_dim=input_dim, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(100, activation='relu'))

    # decoder
    model.add(Dense(300, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(784, activation='sigmoid'))

    return model
