import tensorflow as tf
from tensorflow.keras import layers, models
from matplotlib import pyplot as plt
from tensorflow.python.ops.gen_batch_ops import batch, batch_function

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 25
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
N_CLASSES = 3
VERSION = 1

# TODO: Change this to be a class with methods

def load_dataset():
    # Load the dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "PlantVillage",
        shuffle=True,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size = BATCH_SIZE
    )
    return train_ds

def analyze_dataset(train_ds):
    # Should be three class names
    print(train_ds.class_names)
    print(len(train_ds)) # this is 68 because we have batch size of 32 and total of 2152 images
    for image_batch, label_batch in train_ds:
        print(image_batch.shape) # (32, 256, 256, 3) 32 images of size 256x256 with 3 channels (RGB)
        print(label_batch.numpy()) # this will give you the labels of the images (0, 1, 2)
        plt.imshow(image_batch[0].numpy().astype("uint8"))
        plt.imshow(image_batch[0].numpy().astype("uint8"))
        plt.savefig("output_image.png")
        print("Image saved to output_image.png")
        break

def split_dataset(train_ds, train_size = 0.8, validation_size = 0.1, test_size = 0.1, shuffle = True, shuffle_size = 10000):
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


def resize_and_scale() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.Rescaling(1./255),
        ])

def data_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        ])

def build_cnn(resize_and_scale: tf.keras.Sequential, data_augmentation: tf.keras.Sequential):
    inputs = tf.keras.Input(shape=INPUT_SHAPE)  # Define the input shape
    x = resize_and_scale(inputs)               # Apply resizing and scaling
    x = data_augmentation(x)                   # Apply data augmentation

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
    outputs = layers.Dense(N_CLASSES, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def plot_training_history(history):
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
    dataset = load_dataset()
    train_dataset, validation_dataset, test_dataset = split_dataset(dataset)
    print(len(train_dataset), len(validation_dataset), len(test_dataset))
    # cache the dataset
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

    # Resize and scale the images
    resize_and_scale = resize_and_scale()
    data_augmentation = data_augmentation()

    # Create the model
    model = build_cnn(resize_and_scale, data_augmentation)
    model.summary()

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(train_dataset, validation_data=validation_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        verbose=1)

    # Evaluate the model
    model.evaluate(test_dataset)

    # Save the model evaluation:
    plot_training_history(history)

    # Save the model
    model.export("../models/potato_disease_classifier")
