#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def create_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Define a simple CNN model for MNIST.
    """
    model = models.Sequential()

    # Convolutional block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

def main():
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPUs detected. Using GPU:", gpus)
    else:
        print("No GPU detected, running on CPU.")

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    # Reshape & normalize data
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255.0
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255.0

    # Create the CNN model
    model = create_model()

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    model.fit(
        x_train,
        y_train,
        epochs=1,
        batch_size=128,
        steps_per_epoch=1,  # Limit training to 1 step
        validation_split=0.1
    )

    # Evaluate on test set
    # test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    # print(f"\nTest accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()
