from random import randint

from numpy import array_equal

from attention import AttentionDecoder
from encoding import one_hot_decode, one_hot_encode
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.models import Sequential

# configure problem
n_features = 50
n_timesteps_in = 5
n_timesteps_out = 2


# generate a sequence of random integers
def generate_sequence(length, n_unique):
    return [randint(0, n_unique - 1) for _ in range(length)]


# prepare data for the LSTM
def get_pair(n_in, n_out, n_unique):
    # generate random sequence
    sequence_in = generate_sequence(n_in, n_unique)
    sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in - n_out)]
    # one hot encode
    X = one_hot_encode(sequence_in, n_unique)
    y = one_hot_encode(sequence_out, n_unique)
    # reshape as 3D
    X = X.reshape((1, X.shape[0], X.shape[1]))
    y = y.reshape((1, y.shape[0], y.shape[1]))
    return X, y


if __name__ == "__main__":
    # generate random sequence
    X, y = get_pair(5, 2, 50)
    print(X.shape, y.shape)
    print('X=%s, y=%s' % (one_hot_decode(X[0]), one_hot_decode(y[0])))

    # define model
    model = Sequential()
    model.add(LSTM(150, input_shape=(n_timesteps_in, n_features)))
    model.add(RepeatVector(n_timesteps_in))
    model.add(LSTM(150, return_sequences=True))
    model.add(TimeDistributed(Dense(n_features, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # define model
    model = Sequential()
    model.add(LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=True))
    model.add(AttentionDecoder(150, n_features))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # train LSTM
    for epoch in range(5000):
        # generate new random sequence
        X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
        # fit model for one epoch on this sequence
        model.fit(X, y, epochs=1, verbose=2)

    total, correct = 100, 0
    for _ in range(total):
        X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
        yhat = model.predict(X, verbose=0)
        if array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
            correct += 1
    print('Accuracy: %.2f%%' % (float(correct) / float(total) * 100.0))

    # spot check some examples
    for _ in range(10):
        X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
        yhat = model.predict(X, verbose=0)
        print('Expected:', one_hot_decode(y[0]), 'Predicted', one_hot_decode(yhat[0]))
