import keras
import keras.src.layers as layers
import numpy as np
import tensorflow as tf
from keras.api.datasets import mnist
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def load_and_preprocess_data():
    """Load and preprocess MNIST dataset."""
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    # Normalize and reshape
    train_X = train_X[..., np.newaxis] / 255.0
    test_X = test_X[..., np.newaxis] / 255.0

    # Pad images to 32x32 for LeNet architecture
    train_X = tf.pad(train_X, [[0, 0], [2, 2], [2, 2], [0, 0]]).numpy()
    test_X = tf.pad(test_X, [[0, 0], [2, 2], [2, 2], [0, 0]]).numpy()

    # One-hot encode labels
    train_y = keras.utils.to_categorical(train_y, 10)
    test_y = keras.utils.to_categorical(test_y, 10)

    return train_X, train_y, test_X, test_y


def build_lenet_model(learning_rate=0.001):
    """Build LeNet-5 model architecture."""
    model = keras.Sequential([
        layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 1), padding='same'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),
        layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),
        layers.Flatten(),
        layers.Dense(units=120, activation='relu'),
        layers.Dense(units=84, activation='relu'),
        layers.Dense(units=10, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

    return model


def plot_training_history(history):
    """Plot training metrics."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy plot
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()


def main():
    # Load and preprocess data
    train_X, train_y, test_X, test_y = load_and_preprocess_data()

    # Build model
    model = build_lenet_model(learning_rate=0.001)

    # Train model
    history = model.fit(
        train_X, train_y,
        batch_size=128,
        epochs=20,
        verbose=1,  # Show progress
        validation_data=(test_X, test_y)
    )

    # Evaluate model
    plot_training_history(history)

    # Generate predictions and confusion matrix
    y_pred = np.argmax(model.predict(test_X), axis=1)
    y_true = np.argmax(test_y, axis=1)
    plot_confusion_matrix(y_true, y_pred)

    # Print classification report
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()
