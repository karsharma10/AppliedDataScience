import tensorflow as tf
from tensorflow.keras import layers, models
from matplotlib import pyplot as plt
from tensorflow.python.ops.gen_batch_ops import batch, batch_function

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3

# Load the dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
     batch_size = BATCH_SIZE
)

# Should be three class names
print(train_ds.class_names)
print(len(train_ds)) # this is 68 because we have batch size of 32 and total of 2152 images

for image_batch, label_batch in train_ds:
    print(image_batch.shape) # (32, 256, 256, 3) 32 images of size 256x256 with 3 channels (RGB)
    print(label_batch.numpy()) # this will give you the labels of the images (0, 1, 2)
    break