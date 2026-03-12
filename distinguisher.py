import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation
from tensorflow.keras.layers import Flatten, Dense, Dropout
from sklearn.metrics import accuracy_score


# load csv and skip metadata line
def load_data(path):
    df = pd.read_csv(path, comment="#")
    X = df[["c0l", "c0r", "c1l", "c1r"]].values.astype(np.uint16)
    y = df["label"].values.astype(np.float32)
    return X, y

# convert 4 uint16 words into a 4x16 bit matrix
def convert_to_binary(X):
    n = X.shape[0]
    X_bin = np.zeros((n, 16, 4), dtype=np.float32)

    for i in range(n):
        for j in range(4):
            value = X[i, j]
            for b in range(16):
                X_bin[i, b, j] = (value >> b) & 1

    return X_bin


# simple cnn model
def make_model():
    model = Sequential()

    model.add(Input(shape=(16, 4)))

    model.add(Conv1D(filters=32, kernel_size=1, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv1D(filters=32, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv1D(filters=64, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Flatten())

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


# load train and test files
X_train_words, y_train = load_data("train_nr_4.csv")
X_test_words, y_test = load_data("test_nr_4.csv")

# convert to binary tensor
X_train = convert_to_binary(X_train_words)
X_test = convert_to_binary(X_test_words)

model = make_model()

# train
history = model.fit(X_train, y_train, epochs=10, batch_size=256, validation_data=(X_test, y_test), verbose=1)

# predict
y_prob = model.predict(X_test, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(np.float32)

# scores
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Train accuracy:", model.evaluate(X_train, y_train, verbose=0)[1])
print("Test accuracy:", model.evaluate(X_test, y_test, verbose=0)[1])

# sanity check with random labels
y_train_random = np.random.permutation(y_train)

model_random = make_model()
model_random.fit(X_train, y_train_random, epochs=10, batch_size=256, verbose=0)

y_prob_random = model_random.predict(X_test, verbose=0).ravel()
y_pred_random = (y_prob_random >= 0.5).astype(np.float32)

print("Random-label accuracy:", accuracy_score(y_test, y_pred_random))