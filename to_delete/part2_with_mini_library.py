import torch
import pickle
import numpy as np
import pandas as pd
from part1_nn_lib import MultiLayerNetwork, Trainer
from sklearn.model_selection import GridSearchCV, train_test_split

class Regressor:

    def __init__(self, x, lr=1e-3, bs=16, nb_epoch = 100):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        self.lr = lr
        self.bs = bs
        self.nb_epoch = nb_epoch
        self.network = None

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network and return NumPy arrays.

        - Fill missing numeric values
        - One-hot encode categorical columns
        - Scale numeric features only
        - Ensure training and test have the same columns
        """
        from sklearn.preprocessing import LabelBinarizer, StandardScaler

        # Convert to DataFrame if needed
        x_df = pd.DataFrame(x) if isinstance(x, np.ndarray) else x.copy()
        y_array = np.array(y, dtype=float) if y is not None else None

        # Fill missing numeric values
        if training:
            self.fill_values_ = x_df.mean(numeric_only=True)
        x_df = x_df.fillna(self.fill_values_)

        # Separate numeric and categorical columns
        num_cols = x_df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = x_df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Handle categorical variables
        encoded_dfs = []
        if cat_cols:
            if training:
                self.label_binarizers_ = {}
            for col in cat_cols:
                if training:
                    lb = LabelBinarizer()
                    lb.fit(x_df[col].astype(str))
                    self.label_binarizers_[col] = lb
                else:
                    lb = self.label_binarizers_[col]
                encoded = lb.transform(x_df[col].astype(str))
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=[f"{col}_{cls}" for cls in lb.classes_],
                    index=x_df.index
                )
                encoded_dfs.append(encoded_df)
        x_categorical = pd.concat(encoded_dfs, axis=1) if encoded_dfs else pd.DataFrame(index=x_df.index)

        # Scale numeric features only
        if training:
            self.num_cols_ = num_cols
            self.scaler_ = StandardScaler()
            x_numeric_scaled = pd.DataFrame(
                self.scaler_.fit_transform(x_df[self.num_cols_]),
                columns=self.num_cols_,
                index=x_df.index
            )
        else:
            x_numeric_scaled = pd.DataFrame(
                self.scaler_.transform(x_df.reindex(columns=self.num_cols_, fill_value=0)),
                columns=self.num_cols_,
                index=x_df.index
            )

        # Align one-hot columns with training
        if training:
            self.ohe_cols_ = x_categorical.columns.tolist()
        else:
            x_categorical = x_categorical.reindex(columns=self.ohe_cols_, fill_value=0)

        # Concatenate numeric and categorical features
        x_final = pd.concat([x_numeric_scaled, x_categorical], axis=1)

        # Save CSV only once during training
        if training and not hasattr(self, "_saved_preprocessed_csv"):
            pd.DataFrame(x_final).to_csv("X_preprocessed.csv", index=False)
            if y is not None:
                pd.DataFrame(y_array).to_csv("Y_preprocessed.csv", index=False)
            self._saved_preprocessed_csv = True

        return np.array(x_final, dtype=float), y_array

            

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """
        # Preprocessing
        X, Y = self._preprocessor(x, y=y, training=True)

        # Define input size AFTER preprocessing
        self.input_size = X.shape[1]
        self.output_size = 1

        # Initialize the NN network once the data is preprocessed
        self.network = MultiLayerNetwork(
            input_dim=self.input_size,
            neurons=[32, 16, self.output_size],
            activations=["relu", "relu", "identity"]
        )

        # Training
        self.trainer = Trainer(
            network=self.network,
            batch_size=self.bs,
            nb_epoch=self.nb_epoch,
            learning_rate=self.lr,
            loss_fun="mse",
            shuffle_flag=True
        )
        self.trainer.train(X, Y)
        return self


            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training=False)
        predictions = self.network.forward(X)
        
        return np.array(predictions)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        #To avoid callin predict and RE-process the data
        y_pred = self.network.forward(X)

        # Compute Mean Squared Error (MSE)
        mse = np.mean((Y - y_pred) ** 2)

        # Compute Mean Absolute Error (MAE)
        mae = np.mean(np.abs(Y - y_pred))

        # Compute R-squared (R²)
        ss_res = np.sum((Y - y_pred) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        R2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return mse, mae, R2

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
        


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def perform_hyperparameter_search(X_train, y_train): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################


    NN_model = Regressor(X_train)

    parameter_grid = {
        'lr': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        'bs': [16, 32, 64, 128],
        'epochs':[80, 90, 100]
    }

    param_grid = GridSearchCV(NN_model, parameter_grid)

    param_grid.fit(X_train, y_train)


    return param_grid.best_params_, parameter_grid.best_estimator_, parameter_grid.cv_results

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def gridsearch(X_train, X_val, y_train, y_val):
    best_score = float('inf')
    best_params = None
    best_model = None
#, 3e-4, 1e-3, 3e-3, 1e-2
#, 32, 64, 128
#, 90, 100
    for lr in [1e-4]:
        for bs in [16, 32, 64]:
            for nb_epoch in [80]:
                model = Regressor(X_train, lr=lr, bs=bs, nb_epoch=nb_epoch)
                model.fit(X_train, y_train)
                mse, _, _ = model.score(X_val, y_val)
                if mse < best_score:
                    best_score = mse
                    best_params = {'lr': lr, 'bs': bs, 'nb_epoch': nb_epoch}
                    best_model = model

    return best_params, best_model

def main():
    output_label = "median_house_value"

    # Load data
    data = pd.read_csv("housing.csv")

    # Split input/output
    X = data.drop(columns=[output_label])
    y = data[[output_label]]

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Further split training for validation
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # --- Train best model with gridsearch ---
    best_params, best_regressor = gridsearch(
        X_train_main, X_val, y_train_main, y_val
    )

    # Save model
    save_regressor(best_regressor)
    print('best parameters: ', best_params)
    # --- Metrics ---
    # Validation metrics (optional)
    val_mse, val_mae, val_r2 = best_regressor.score(X_val, y_val)
    print(f"Validation - MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")

    # Test metrics (final evaluation)
    test_mse, test_mae, test_r2 = best_regressor.score(X_test, y_test)
    print(f"Test       - MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")


if __name__ == "__main__":
    main()
