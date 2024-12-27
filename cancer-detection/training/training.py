import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from sklearn.ensemble import RandomForestClassifier



if __name__ == "__main__":
    df = pd.read_csv("breast-cancer.csv")

    print(df.head())

    numerical_columns = df.select_dtypes(include=["number"]).columns

    print(len(numerical_columns))

    # Target label (y)
    y_label = df['diagnosis']

    # Feature columns (X)
    x_labels = df.drop(columns=['diagnosis'])
    x_labels.drop(columns=["id"])

    # Check for missing values
    print(x_labels.isnull().sum())

    # Train a simple RandomForest to identify important features
    model = RandomForestClassifier()
    model.fit(x_labels, y_label)

    # Get feature importance
    importance = pd.Series(model.feature_importances_, index=x_labels.columns)
    important_features = importance.nlargest(10).index  # Top 10 features
    print(important_features)
    x_labels = x_labels[important_features]




class NNCancerClassification:
    def __init__(self):
        self.input_shape = (30,) # we have 30 features to use
        self.norm_layer = layers.Normalization(axis = -1)

    def normalize_input(self, data):
        return self.norm_layer.adapt(data)

    def build_nn(self):
        input_layer = Input(shape=self.input_shape)
        x = self.normalize_input(input_layer)