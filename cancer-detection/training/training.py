from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Normalization
from tensorflow.keras.models import Model
import pandas as pd
from sklearn.model_selection import train_test_split



class NNCancerClassification:
    def __init__(self):
        self.input_shape = (30,)  # We have 30 features to use
        self.norm_layer = Normalization(axis=-1)  # Define normalization layer

    def normalize_input(self, x_train):
        """
        Adapt the normalization layer to the training data.
        Convert x_train to NumPy if it's a DataFrame.
        """
        if isinstance(x_train, pd.DataFrame):  # Check if input is a Pandas DataFrame
            x_train = x_train.to_numpy()  # Convert to NumPy array
        self.norm_layer.adapt(x_train)  # Compute mean and variance from x_train

    def build_nn(self):
        """
        Build the neural network model for cancer classification.
        """
        # Input layer
        input_layer = Input(shape=self.input_shape)

        # Apply normalization
        x = self.norm_layer(input_layer)

        # Add dense layers with BatchNormalization and Dropout
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        # Output layer for binary classification
        output_layer = Dense(1, activation='sigmoid')(x)

        # Define the model
        model = Model(inputs=input_layer, outputs=output_layer)

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model


if __name__ == "__main__":
    df = pd.read_csv("breast-cancer.csv")
    # Separate features and target
    y_label = df['diagnosis'].map({'M': 1, 'B': 0})  # Map diagnosis to binary labels
    x_labels = df.drop(columns=['diagnosis', 'id'])  # Drop target and unnecessary columns

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(x_labels, y_label, test_size=0.2, random_state=42)

    # Instantiate the class
    newNN = NNCancerClassification()

    # Adapt normalization layer with training data (converted to NumPy)
    newNN.normalize_input(X_train)

    # Build the model
    model = newNN.build_nn()

    # Train the model
    model.fit(X_train.to_numpy(), y_train, epochs=25, batch_size=16, validation_data=(X_test.to_numpy(), y_test))