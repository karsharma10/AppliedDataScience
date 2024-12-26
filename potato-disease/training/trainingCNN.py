import tensorflow as tf
from tensorflow.keras import layers, models
from matplotlib import pyplot as plt


class DiseaseClassificationCNN:
    def __init__(self, image_size, batch_size, epochs):
        """
        Initialize the CNN model for disease classification
        :param image_size:
        :param batch_size:
        :param epochs:
        """
        self.dataset = None
        self.image_size = image_size
        self.batch_size = batch_size
        self.channel_size = 3 # const for images (R,G, B)
        self.epochs = epochs
        self.input_shape = (self.image_size, self.image_size, self.channel_size)
        self.n_classes = 3

    def load_dataset(self):
        """
        function will load the image dataset from directory, which is split up into
        classification folders.
        :return: None
        """
        # Load the dataset
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            "PlantVillage",
            shuffle=True,
            image_size=(self.image_size, self.image_size),
            batch_size = self.batch_size
        )
        self.dataset = train_ds

    def _train_validation_split(self, train_ds, train_size = 0.8, validation_size = 0.1, test_size = 0.1,
                                shuffle = True, shuffle_size = 10000):
        """
        Function will split the dataset into training, validation, and testing datasets
        :param train_ds:
        :param train_size:
        :param validation_size:
        :param test_size:
        :param shuffle:
        :param shuffle_size:
        :return: train_dataset, validation_dataset, test_dataset
        """
        assert train_ds is not None, "Dataset is not initialized, please call load dataset"
        # Split the dataset into training, validation, and testing datasets
        total_size = len(train_ds)  # Total number of batches in the dataset

        if shuffle:
            train_ds = train_ds.shuffle(shuffle_size, seed = 12)

        train_count = int(total_size * train_size)
        test_count = int(total_size * test_size)

        train_dataset = train_ds.take(train_count)
        remaining_dataset = train_ds.skip(train_count)

        test_dataset = remaining_dataset.take(test_count)
        validation_dataset = remaining_dataset.skip(test_count)

        return train_dataset, validation_dataset, test_dataset

    def resize_and_scale(self) -> tf.keras.Sequential:
        """
        Function will resize and scale the image
        :return:
        """
        return tf.keras.Sequential([
            layers.Resizing(self.image_size, self.image_size),
            layers.Rescaling(1./255),
        ])
    @staticmethod
    def data_augmentation() -> tf.keras.Sequential:
        """
        Function will apply data augmentation to the image
        :return:
        """
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
        ])

    def _build_cnn(self):
        """
        Function will build the CNN model
        :return:
        """
        inputs = tf.keras.Input(shape=self.input_shape)  # Define the input shape
        x = self.resize_and_scale()(inputs)               # Apply resizing and scaling
        x = self.data_augmentation()(x)                   # Apply data augmentation

        # Add convolutional and pooling layers
        x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Add the fully connected layers
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(self.n_classes, activation='softmax')(x)

        # Create the model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def train_model(self):
        """
        Function will train the CNN model
        :return:
        """
        # Step 1: build the dataset and split dataset
        self.load_dataset()
        train_dataset, validation_dataset, test_dataset = self._train_validation_split(self.dataset)

        # Step 2: Cache the dataset split
        train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
        test_dataset = test_dataset.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

        # Step 3: Build the CNN
        cnn_model = self._build_cnn()
        cnn_model.summary()

        # Step 4: Compile the Model
        cnn_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Step 5: Train the Model:
        history = cnn_model.fit(train_dataset, validation_data=validation_dataset, batch_size=self.batch_size,
                                epochs=self.epochs, verbose=1)

        # Step 6: Evaluate the model
        cnn_model.evaluate(test_dataset)

        # Step 7: Save the model evaluation:
        plot_training_history(history)

        # Step 8: Save the model
        cnn_model.export("../models/potato_disease_classifier")

def plot_training_history(history):
    """
    Function will plot the training and validation accuracy
    :param history:
    :return:
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')  # Blue dots and line
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')  # Red dots and line
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('training_validation_accuracy.png')  # Save the plot
    print("Plot saved as 'training_validation_accuracy.png'")



if __name__ == "__main__":
    # Initialize the CNN model
    potato_classifier = DiseaseClassificationCNN(image_size=256, batch_size=32, epochs=25)
    # Train the model
    potato_classifier.train_model()