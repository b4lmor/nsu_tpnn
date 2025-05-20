from tqdm import tqdm

from lab3.hard.networks import *
from lab3.lazy.main import load_and_preprocess_data, create_data_loaders, plot_comparison, print_comparison_table, \
    find_best_model

TEST_SIZE = 0.8
NUM_CLASSES = 3
HIDDEN_SIZE = 32
BATCH_SIZE = 64
SEQUENCE_LENGTH = 4
NUM_EPOCHS = 20


X_train, X_test, y_train, y_test = load_and_preprocess_data()
train_loader, test_loader = create_data_loaders(
    X_train, X_test, y_train, y_test, BATCH_SIZE, SEQUENCE_LENGTH)

model = RNN_Classifier(X_train.shape[1], NUM_CLASSES, HIDDEN_SIZE, RNN)


def train_model(model, train_loader, test_loader, model_name, num_epochs=NUM_EPOCHS):
    optimizer = SGD(model, lr=0.01, momentum=0.9)
    criterion = CrossEntropyLoss()

    RNN_train_losses, RNN_train_accuracies = [], []
    RNN_test_losses, RNN_test_accuracies = [], []
    for _ in tqdm(range(1, num_epochs + 1), desc=f"Training {model_name}"):
        RNN_train_loss, RNN_train_accuracy = 0.0, 0.0
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.detach().numpy()
            y_batch = y_batch.detach().numpy()
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            model.backward(X_batch, criterion.backward(predictions, y_batch))
            optimizer.step()
            cur_accuracy = np.sum(predictions.argmax(axis=1) == y_batch)
            RNN_train_loss += loss * X_batch.shape[0]
            RNN_train_accuracy += cur_accuracy
        RNN_test_loss, RNN_test_accuracy = 0.0, 0.0
        model.eval()
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.detach().numpy()
            y_batch = y_batch.detach().numpy()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            RNN_test_loss += loss.item() * X_batch.shape[0]
            RNN_test_accuracy += np.sum(predictions.argmax(axis=1) == y_batch)
        RNN_train_loss /= len(train_loader.dataset)
        RNN_test_loss /= len(test_loader.dataset)
        RNN_train_accuracy /= len(train_loader.dataset)
        RNN_test_accuracy /= len(test_loader.dataset)
        RNN_train_losses += [RNN_train_loss]
        RNN_train_accuracies += [RNN_train_accuracy]
        RNN_test_losses += [RNN_test_loss]
        RNN_test_accuracies += [RNN_test_accuracy]
    return {
        'train_losses': RNN_train_losses,
        'test_losses': RNN_test_losses,
        'train_accuracies': RNN_train_accuracies,
        'test_accuracies': RNN_test_accuracies
    }


def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    train_loader, test_loader = create_data_loaders(
        X_train, X_test, y_train, y_test, BATCH_SIZE, SEQUENCE_LENGTH)

    models = {
        'RNN': RNN_Classifier(X_train.shape[1], NUM_CLASSES, HIDDEN_SIZE, RNN),
        'GRU': RNN_Classifier(X_train.shape[1], NUM_CLASSES, HIDDEN_SIZE, GRU),
        'LSTM': RNN_Classifier(X_train.shape[1], NUM_CLASSES, HIDDEN_SIZE, LSTM)
    }

    results = {}
    for model_name, model in models.items():
        results[model_name] = train_model(model, train_loader, test_loader, model_name, NUM_EPOCHS)

    plot_comparison(results)

    print_comparison_table(results)

    find_best_model(results)


if __name__ == "__main__":
    main()
