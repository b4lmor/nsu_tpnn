import numpy as np
from keras.api.datasets import mnist
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from lab4.hard.lenet import Lenet5, cross_entropy_loss, softmax_cross_entropy_backward


def evaluate_model(model, test_images, test_labels) -> tuple[float, float]:
    """Evaluate model performance on test data."""

    total_loss = 0.0
    correct_predictions = 0

    for image, true_label in zip(test_images, test_labels):
        # Forward pass
        probabilities = model.forward(image)

        # Calculate loss
        loss = cross_entropy_loss(probabilities, true_label)
        total_loss += loss

        # Calculate accuracy
        predicted_class = np.argmax(probabilities.ravel())
        if predicted_class == true_label:
            correct_predictions += 1

    average_loss = total_loss / len(test_images)
    accuracy_percentage = (correct_predictions / len(test_images)) * 100

    return average_loss, accuracy_percentage


def preprocess_data():
    """Load and preprocess MNIST dataset."""

    # Load dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize pixel values
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # Add channel dimension (N, 28, 28) -> (N, 1, 28, 28)
    train_images = np.expand_dims(train_images, axis=1)
    test_images = np.expand_dims(test_images, axis=1)

    # Pad images to 32x32 (2 pixels on each side)
    padding = ((0, 0), (0, 0), (2, 2), (2, 2))
    train_images = np.pad(train_images, padding, mode='constant')
    test_images = np.pad(test_images, padding, mode='constant')

    # Use smaller subset for demonstration
    train_images = train_images[:100]
    test_images = test_images[:500]
    train_labels = train_labels[:100]
    test_labels = test_labels[:500]

    return train_images, train_labels, test_images, test_labels


def train_model(model, train_images, train_labels, test_images, test_labels, num_epochs=10, learning_rate=0.01):
    """Train the model and track performance metrics."""

    loss_history = []
    accuracy_history = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}:")
        for image, true_label in tqdm(zip(train_images, train_labels),
                                      total=len(train_images),
                                      desc="Training"):
            probabilities = model.forward(image)

            cross_entropy_loss(probabilities, true_label)
            gradient = softmax_cross_entropy_backward(probabilities, true_label)
            model.backward(gradient, learning_rate)

        epoch_loss, epoch_accuracy = evaluate_model(model, test_images, test_labels)
        loss_history.append(epoch_loss)
        accuracy_history.append(epoch_accuracy)

        print(f"Epoch {epoch} metrics:")
        print(f"  Loss: {epoch_loss:.4f}")
        print(f"  Accuracy: {epoch_accuracy:.2f}%")
        print("-" * 40)

    return loss_history, accuracy_history


def plot_metrics(loss_history, accuracy_history):
    """Plot training loss and accuracy metrics."""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Loss plot
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(range(1, len(loss_history) + 1), loss_history, color='tab:red', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(loc='upper right')
    ax1.set_title("Validation Loss")

    # Accuracy plot
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)', color='tab:blue')
    ax2.plot(range(1, len(accuracy_history) + 1), accuracy_history,
             color='tab:blue', label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.legend(loc='upper left')
    ax2.set_title("Validation Accuracy")

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, test_images, test_labels):
    """Generate and plot confusion matrix."""

    true_classes = []
    predicted_classes = []

    for image, true_label in tqdm(zip(test_images, test_labels), total=len(test_images)):
        probabilities = model.forward(image)
        predicted_class = np.argmax(probabilities)
        predicted_classes.append(predicted_class)
        true_classes.append(true_label)

    cm = confusion_matrix(true_classes, predicted_classes)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


def main():
    # Load and preprocess data
    train_images, train_labels, test_images, test_labels = preprocess_data()

    # Initialize model
    model = Lenet5()

    # Train model
    loss_history, accuracy_history = train_model(
        model,
        train_images,
        train_labels,
        test_images,
        test_labels,
        num_epochs=10,
        learning_rate=0.01
    )

    # Visualize results
    plot_metrics(loss_history, accuracy_history)
    plot_confusion_matrix(model, test_images, test_labels)


if __name__ == "__main__":
    main()
