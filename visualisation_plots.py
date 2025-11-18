import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler


class Regressor:
    def __init__(self, x, lr=1e-4, bs=64, nb_epoch=150):
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.lb = LabelBinarizer()


        # Determine input and output sizes
        X_dummy, _ = self._preprocessor(x, training=True)
        self.input_size = X_dummy.shape[1]
        self.output_size = 1

        self.lr = lr
        self.bs = bs
        self.nb_epoch = nb_epoch

        self.model = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_size)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.train_mse_history = []

    def _preprocessor(self, x, y=None, training=False):
        x = x.copy()
        num_cols = x.select_dtypes(include=[np.number]).columns.tolist()

        if training:
            x[num_cols] = x[num_cols].fillna(x[num_cols].median())
            y = y.fillna(y.median()) if y is not None else None
        else:
            x[num_cols] = x[num_cols].fillna(0)
            y = y.fillna(0) if y is not None else None

        cat_col = 'ocean_proximity'
        if training:
            x[cat_col] = x[cat_col].fillna(x[cat_col].mode()[0])
            self.lb.fit(x[cat_col].astype(str))
        else:
            x[cat_col] = x[cat_col].fillna("None")
            known = set(self.lb.classes_)
            x[cat_col] = x[cat_col].apply(lambda v: v if v in known else "None")

        encoded = self.lb.transform(x[cat_col].astype(str))
        x = x.drop(columns=[cat_col])

        if training:
            x = self.x_scaler.fit_transform(x)
            y = self.y_scaler.fit_transform(y) if y is not None else None
        else:
            x = self.x_scaler.transform(x)
            y = self.y_scaler.transform(y) if y is not None else None

        X = torch.tensor(x, dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        return X, Y

    def fit(self, x, y):
        X, Y = self._preprocessor(x, y=y, training=True)
        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.bs, shuffle=True)

        self.model.train()
        self.train_mse_history = []

        for epoch in range(self.nb_epoch):
            epoch_loss = 0
            for xb, yb in loader:
                preds = self.model(xb)
                loss = self.loss_fn(preds, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            # Compute unscaled MSE for this epoch
            with torch.no_grad():
                preds_full = self.model(X).numpy()
                preds_unscaled = self.y_scaler.inverse_transform(preds_full)
                y_unscaled = self.y_scaler.inverse_transform(Y.numpy())
                mse_epoch = float(np.mean((y_unscaled - preds_unscaled) ** 2))

            self.train_mse_history.append(mse_epoch)
            print(f"Epoch {epoch+1}/{self.nb_epoch}, Unscaled Loss: {mse_epoch:.6f}")

        return self

    def predict(self, x):
        X, _ = self._preprocessor(x, training=False)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X).numpy()
        preds = self.y_scaler.inverse_transform(preds)
        return preds

    def score(self, x, y):
        X, _ = self._preprocessor(x, y=y, training=False)
        self.model.eval()
        with torch.no_grad():
            preds_scaled = self.model(X).numpy()
            preds = self.y_scaler.inverse_transform(preds_scaled)

        y_array = y.values if isinstance(y, pd.DataFrame) else y
        mse = mean_squared_error(y_array, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_array, preds)
        return {"MSE": mse, "RMSE": rmse, "R2": r2}


def save_regressor(trained_model, path='part2_model.pickle'):
    with open(path, 'wb') as target:
        pickle.dump(trained_model, target)
    print(f"Saved model in {path}")


def load_regressor(path='part2_model.pickle'):
    with open(path, 'rb') as target:
        trained_model = pickle.load(target)
    print(f"Loaded model from {path}")
    return trained_model


def perform_hyperparameter_search(X_train, X_val, y_train, y_val):
    best_score = float('inf')
    best_params = None
    best_model = None
    all_histories = []
    val_metrics_list = []

    for lr in [0.02, 0.003]:
        for bs in [64]:
                print(f"Training with lr={lr}, bs={bs}")

                model = Regressor(X_train, lr=lr, bs=bs)
                model.fit(X_train, y_train)

                # Predict on validation set
                y_pred = model.predict(X_val).flatten()
                y_true = y_val.values.flatten()

                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, y_pred)

                label = f"lr={lr}, bs={bs}"
                val_metrics_list.append({
                    "configuration": label,
                    "MSE": mse,
                    "RMSE": rmse,
                    "R2": r2
                })

                print(f"Validation metrics for {label}: MSE={mse:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")
                all_histories.append((label, model.train_mse_history))

                if mse < best_score:
                    best_score = mse
                    best_params = {'lr': lr, 'bs': bs}
                    best_model = model

    # Plot training curves
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    for label, hist in all_histories:
        plt.plot(hist, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE")
    plt.title("Training MSE per Epoch for All Hyperparameter Configurations")
    plt.legend()
    plt.savefig("training_mse_curves.png")
    plt.close()

    return best_params, best_model, val_metrics_list


def example_main():
    output_label = "median_house_value"
    data = pd.read_csv("housing.csv")

    X = data.drop(columns=[output_label])
    y = data[[output_label]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    best_params, best_model, val_metrics_list = perform_hyperparameter_search(X_train, X_val, y_train, y_val)

    print("\nValidation metrics for all configurations:")
    for metrics in val_metrics_list:
        print(metrics)

    print(f"\nBest hyperparameters found: {best_params}")

    regressor = best_model
    save_regressor(regressor)

    # Compute metrics on train and test
    train_scores = regressor.score(X_train, y_train)
    test_scores = regressor.score(X_test, y_test)

    print(f"\nTrain scores: {train_scores}")
    print(f"Test scores: {test_scores}")


if __name__ == "__main__":
    example_main()
