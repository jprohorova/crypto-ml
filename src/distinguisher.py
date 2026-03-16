import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation
from tensorflow.keras.layers import Flatten, Dense, Dropout
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def ensure_results_dir():
    path = "results"
    os.makedirs(path, exist_ok=True)
    return path


def load_data(path):
    df = pd.read_csv(path, comment="#")
    X = df[["c0l", "c0r", "c1l", "c1r"]].values.astype(np.uint16)
    y = df["label"].values.astype(np.float32)
    return X, y


def convert_to_binary(X):
    n_samples = X.shape[0]
    X_bin = np.zeros((n_samples, 16, 4), dtype=np.float32)

    for i in range(n_samples):
        for j in range(4):
            value = X[i, j]
            for b in range(16):
                X_bin[i, b, j] = (value >> b) & 1  # extract single bit value

    return X_bin


def get_data_for_rounds(rounds, data_dir="data"):
    train_path = os.path.join(data_dir, f"train_nr_{rounds}.csv")
    test_path = os.path.join(data_dir, f"test_nr_{rounds}.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing file: {test_path}")

    X_train_words, y_train = load_data(train_path)
    X_test_words, y_test = load_data(test_path)

    X_train = convert_to_binary(X_train_words)
    X_test = convert_to_binary(X_test_words)

    return X_train, y_train, X_test, y_test, train_path, test_path


def make_model():
    model = Sequential()

    model.add(Input(shape=(16, 4)))

    model.add(Conv1D(filters=32, kernel_size=1, padding="same")) # learn local bit relations
    model.add(BatchNormalization()) # stabilize hidden activation scales
    model.add(Activation("relu"))

    model.add(Conv1D(filters=32, kernel_size=3, padding="same")) # capture short structured patterns
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv1D(filters=64, kernel_size=3, padding="same")) # increase feature extraction depth
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Flatten())  # convert feature maps vector
    model.add(Dense(128, activation="relu"))  # learn nonlinear decision boundary
    model.add(Dropout(0.3))  # reduce memorization on train
    model.add(Dense(1, activation="sigmoid"))  # output binary class probability

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def plot_training_history(history, rounds):
    results_dir = ensure_results_dir()

    plt.figure()
    plt.plot(history.history["accuracy"], label="Train accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"SPECK {rounds} rounds - accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"training_accuracy_r{rounds}.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"SPECK {rounds} rounds - loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"training_loss_r{rounds}.png"), dpi=300)
    plt.close()


def plot_prediction_histogram(y_prob, y_true, rounds):
    results_dir = ensure_results_dir()

    y_prob = np.asarray(y_prob).ravel()
    y_true = np.asarray(y_true)

    real_scores = y_prob[y_true == 1]
    random_scores = y_prob[y_true == 0]

    plt.figure()
    plt.hist(random_scores, bins=40, alpha=0.6, label="Random pairs")
    plt.hist(real_scores, bins=40, alpha=0.6, label="Real pairs")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title(f"Prediction distribution - SPECK {rounds} rounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"prediction_distribution_r{rounds}.png"), dpi=300)
    plt.close()

def plot_conf_matrix(y_true, y_pred, rounds):
    results_dir = ensure_results_dir()

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Random", "Real"])

    fig, ax = plt.subplots()
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix - SPECK {rounds} rounds")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"confusion_matrix_r{rounds}.png"), dpi=300)
    plt.close(fig)


def plot_baseline_comparison(test_acc, random_acc, rounds):
    results_dir = ensure_results_dir()

    plt.figure()
    plt.bar(["Model", "Random labels"], [test_acc, random_acc])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(f"Model vs random-label baseline - SPECK {rounds} rounds")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"baseline_comparison_r{rounds}.png"), dpi=300)
    plt.close()


def plot_accuracy_vs_rounds(results):
    results_dir = ensure_results_dir()

    rounds = sorted(results.keys())
    test_acc = [results[r]["test_acc"] for r in rounds]
    random_acc = [results[r]["random_acc"] for r in rounds]

    plt.figure()
    plt.plot(rounds, test_acc, marker="o", label="Test accuracy")
    plt.plot(rounds, random_acc, marker="o", label="Random-label accuracy")
    plt.axhline(0.5, linestyle="--", label="Chance level")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.title("Neural distinguisher accuracy vs SPECK rounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "accuracy_vs_rounds.png"), dpi=300)
    plt.close()


def save_results(results):
    results_dir = ensure_results_dir()

    clean_results = {}
    for r, res in results.items():
        clean_results[str(r)] = {
            "train_acc": float(res["train_acc"]),
            "test_acc": float(res["test_acc"]),
            "random_acc": float(res["random_acc"]),
        }

    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(clean_results, f, indent=4)


def run_one_round(rounds, epochs=10, batch_size=256, data_dir="data", seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_train, y_train, X_test, y_test, train_path, test_path = get_data_for_rounds(rounds, data_dir=data_dir)

    model = make_model()
    history = model.fit(X_train,y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
        verbose=1)
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(np.float32) # apply binary decision threshold

    train_acc = float(model.evaluate(X_train, y_train, verbose=0)[1])
    test_acc = float(model.evaluate(X_test, y_test, verbose=0)[1])

    y_train_random = np.random.permutation(y_train) # destroy true label structure
    model_random = make_model()
    model_random.fit(X_train, y_train_random, epochs=epochs, batch_size=batch_size, verbose=0)

    y_prob_random = model_random.predict(X_test, verbose=0).ravel()
    y_pred_random = (y_prob_random >= 0.5).astype(np.float32)
    random_acc = float(accuracy_score(y_test, y_pred_random))

    final_acc = float(accuracy_score(y_test, y_pred))

    print(f"Rounds: {rounds}")
    print(f"Train file: {train_path}")
    print(f"Test file: {test_path}")
    print(f"Accuracy: {final_acc:.4f}")
    print(f"Train accuracy: {train_acc}")
    print(f"Test accuracy: {test_acc}")
    print(f"Random-label accuracy: {random_acc}")

    return {
        "history": history,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "random_acc": random_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, help="Run one experiment for one number of rounds")
    parser.add_argument("--all-rounds", nargs="+", type=int, help="Run several rounds and compare them")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--plots", action="store_true", help="Save per-round plots")
    args = parser.parse_args()

    if args.rounds is not None:
        result = run_one_round(
            rounds=args.rounds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            data_dir=args.data_dir,
        )

        if args.plots:
            plot_training_history(result["history"], args.rounds)
            plot_prediction_histogram(result["y_prob"], result["y_test"], args.rounds)
            plot_conf_matrix(result["y_test"], result["y_pred"], args.rounds)
            plot_baseline_comparison(result["test_acc"], result["random_acc"], args.rounds)

            save_results({
                args.rounds: {
                    "train_acc": result["train_acc"],
                    "test_acc": result["test_acc"],
                    "random_acc": result["random_acc"],
                }
            })

    elif args.all_rounds is not None:
        results = {}

        for r in args.all_rounds:
            print("\n" + "=" * 50)
            print(f"Running round {r}")
            print("=" * 50)

            result = run_one_round(
                rounds=r,
                epochs=args.epochs,
                batch_size=args.batch_size,
                data_dir=args.data_dir,
            )

            results[r] = result

            plot_training_history(result["history"], r)
            plot_prediction_histogram(result["y_prob"], result["y_test"], r)
            plot_conf_matrix(result["y_test"], result["y_pred"], r)
            plot_baseline_comparison(result["test_acc"], result["random_acc"], r)

        plot_accuracy_vs_rounds(results)
        save_results(results)

    else:
        raise SystemExit("Use --rounds N --plots or --all-rounds 4 5 6 7 8")


if __name__ == "__main__":
    main()