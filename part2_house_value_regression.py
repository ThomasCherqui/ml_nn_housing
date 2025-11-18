import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler


class Regressor:

    def __init__(self, x, lr=1e-4, bs=64,nb_epoch = 80):
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
        # initialise the standard scaler
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.lb = LabelBinarizer()
        # Determine input size from the dataframe
        self.input_size = 13
        self.output_size = 1  # regression output
        
        # Hyperparameters of the NN (default values if None)
        self.lr = lr
        self.bs = bs
        self.nb_epoch = nb_epoch
        
        # Initialisation for the feedforward neural network
        self.model = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_size)
        )
        
        # Initiliase the optimizer and the loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss() #it is a regression
        
        #

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Separate the columns (numerical and categorical)
        num_cols = x.select_dtypes(include=[np.number]).columns.tolist()

        # Fill missing numeric values with the median value if training, with zero if not
        if training:
            x[num_cols] = x[num_cols].fillna(x[num_cols].median())
            y = y.fillna(y.median()) if y is not None else None
            
        else:
            x[num_cols] = x[num_cols].fillna(0)
            y = y.fillna(0) if y is not None else None

        # Fill missing categorical values (cat_col) with the None value
        # One-hot encoding for the categorical values if it's training - and if it is not apply the same mapping      
        cat_col = 'ocean_proximity'
        if training:
            x[cat_col] = x[cat_col].fillna(x[cat_col].mode()[0])
            # Fit encoder
            self.lb.fit(x[cat_col].astype(str))
        else:
            # Replace missing values
            x[cat_col] = x[cat_col].fillna("None")
            # Replace unseen values
            known = set(self.lb.classes_)
            x[cat_col] = x[cat_col].apply(lambda v: v if v in known else "None")

        encoded = self.lb.transform(x[cat_col].astype(str))
        x = x.drop(columns=[cat_col])
        
        # Scale the X and Y - keep track of the Scaler for the post-training preprocessing
        if training:
            x = self.x_scaler.fit_transform(x)
            y = self.y_scaler.fit_transform(y) if y is not None else None
        
        else:
            x = self.x_scaler.transform(x)
            y = self.y_scaler.transform(y) if y is not None else None
        
        #Concatenate the scaled numerical columns and the encoded categorical columns
        x = np.concatenate([x, encoded], axis=1)
        
        #Convert the DataFrames to torch.tensor - if y empty, return None
        X = torch.tensor(x, dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        
        return X,Y
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
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

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################


        X, Y = self._preprocessor(x, y, training=True)
        dataset = torch.utils.data.TensorDataset(X, Y)
        # The DataLoader create “minibatches” and reshuffle the data at every epoch to reduce model overfitting
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.bs, shuffle=True)

        self.model.train()
        for epoch in range(self.nb_epoch):
            epoch_loss = 0
            for xb, yb in loader:
                # forward
                preds = self.model(xb)
                loss = self.loss_fn(preds, yb)
                
                # backward & optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                #keep track of the loss for each epoch
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.nb_epoch}, Loss: {epoch_loss/len(loader):.6f}")
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
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

        X, _ = self._preprocessor(x, training = False) # Do not forget
        
        #"eval" mode in Pytorch = stop the dropout and batch normalization layers when evaluating and not training
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X)
        
        #Re-scale the predictions using the correct StandardScaler()
        preds = preds.numpy()
        preds = self.y_scaler.inverse_transform(preds)
        return preds

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

        X, _ = self._preprocessor(x, y=None, training = False) # Do not forget
        # Set to eval mode
        self.model.eval()
        
        # Compute MSE
        with torch.no_grad():
            preds = self.model(X)
            # Convert to numpy
            preds = preds.numpy()
            y = y.to_numpy()

            # Inverse-transform
            preds = self.y_scaler.inverse_transform(preds)

            mse = np.mean((y - preds)**2)
            return mse
    
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


def perform_hyperparameter_search(X_train, X_val, y_train, y_val):
    best_score = float('inf')
    best_params = None
    best_model = None

    for lr in [2e-2,3e-3]:
        for bs in [64]:
            for nb_epoch in [500]:
                print(f"Training with lr={lr}, bs={bs}, nb_epoch={nb_epoch}")
                model = Regressor(X_train, lr=lr, bs=bs, nb_epoch=nb_epoch)
                model.fit(X_train, y_train)
                mse= model.score(X_val, y_val)
                if mse < best_score:
                    best_score = mse
                    best_params = {'lr': lr, 'bs': bs, 'nb_epoch': nb_epoch}
                    best_model = model

    return best_params, best_model

def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    X = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    # Train/ val/ test split
    X_train, X_test, y_train, y_test = train_test_split(
       X , y, test_size=0.1, random_state=42
    ) 
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # Find best hyperparameters
    best_params, best_model = perform_hyperparameter_search(X_train, X_val, y_train, y_val)
    print(f"Best hyperparameters found: {best_params}")
    regressor = best_model
    save_regressor(regressor)

    # Evaluate on train and test data
    mse_train = np.sqrt(regressor.score(X_train, y_train))
    mse_test = np.sqrt(regressor.score(X_test, y_test))

    print(f"\nTrain RMSE: {mse_train:.4f}")
    print(f"Test RMSE: {mse_test:.4f}\n")


if __name__ == "__main__":
    example_main()