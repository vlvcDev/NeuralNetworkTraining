import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import boston_housing
from keras import layers

import matplotlib.pyplot as plt

SEED_VALUE = 42

np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

# Loading dataset
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape)
print("\n")
print("Input features: ", x_train[0])
print("\n")
print("Target: ", y_train[0])

# Extract features from dataset
boston_features = { "Average Number of Rooms": 5, }

X_train_1d = x_train[:, boston_features["Average Number of Rooms"]]
print(X_train_1d.shape)

X_test_1d = x_test[:, boston_features["Average Number of Rooms"]]

plt.figure(figsize=(15, 5))

plt.xlabel("Average Number of Rooms")
plt.ylabel("Median House Price")
plt.grid("on")
plt.scatter(X_train_1d[:], y_train, color="green", alpha=0.5)

model = Sequential()

# Define a single neuron model
model.add(Dense(units=1, input_shape=(1,)))
model.summary()

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.005), loss="mse")

# Train the model
history = model.fit(X_train_1d, y_train, epochs=600, batch_size=32, validation_split=0.3)

def plot_loss(history):
    plt.figure(figsize=(20,5))
    plt.plot(history.history['loss'], 'g', label='Training Loss')
    plt.plot(history.history['val_loss'], 'b', label='Validation Loss')
    plt.xlim([0, 100])
    plt.ylim([0, 300])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss(history)


# Make a prediction
x = [3, 4, 5, 6, 7]
x = np.array(x).reshape(-1, 1)
y_pred = model.predict(x)
for idx in range(len(x)):
    print(f"Predicted price of a home with {x[idx]} rooms: ${int(y_pred[idx] * 10) / 10}K")

# plot the model and data
x = np.linspace(3,9,10)

y = model.predict(x)

def plot_data(x_data, y_data, x, y, title=None):
    plt.figure(figsize=(15, 5))
    plt.scatter(x_data, y_data, label='Ground Truth', color='green', alpha=0.5)
    plt.plot(x, y, color='k', label='Model Predictions')
    plt.xlim([3,9])
    plt.ylim([0,60])
    plt.xlabel('Average Number of Rooms')
    plt.ylabel('Price [$K]')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

plot_data(X_train_1d, y_train, x, y, title='Training Dataset')
plot_data(X_test_1d, y_test, x, y, title='Test Dataset')
