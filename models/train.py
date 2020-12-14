import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow.keras.layers.experimental.preprocessing as preprocessing
import pandas as pd

ds_train_ = image_dataset_from_directory(
    './data/train',
    labels='inferred',
    label_mode='binary',
    image_size=[64, 64],
    interpolation='nearest',
    batch_size=3,
    shuffle=True,
)

def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = (
    ds_train_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

# augment = keras.Sequential([
#     # preprocessing.RandomContrast(factor=0.5),
#     # preprocessing.RandomFlip(mode='horizontal'), # meaning, left-to-right
#     # preprocessing.RandomFlip(mode='vertical'), # meaning, top-to-bottom
#     # preprocessing.RandomWidth(factor=0.15), # horizontal stretch
#     # preprocessing.RandomRotation(factor=0.20),
#     # preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),

#     preprocessing.RandomContrast(factor=0.10),
#     preprocessing.RandomFlip(mode='horizontal'),
#     preprocessing.RandomRotation(factor=0.10),
# ])


# ex = next(iter(ds_train.unbatch().map(lambda x, y: x).batch(1)))

# plt.figure(figsize=(10,10))
# for i in range(16):
#     image = augment(ex, training=True)
#     plt.subplot(4, 4, i+1)
#     plt.imshow(tf.squeeze(image))
#     plt.axis('off')
# plt.show()

model = keras.Sequential([
    layers.InputLayer(input_shape=[64, 64, 3]),
    
    # preprocessing.RandomContrast(factor=0.10),
    # preprocessing.RandomFlip(mode='horizontal'),
    # preprocessing.RandomRotation(factor=0.10),
    
    # Block One
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # # Block Two
    # layers.BatchNormalization(renorm=True),
    # layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    # layers.MaxPool2D(),

    # # Block Three
    # layers.BatchNormalization(renorm=True),
    # layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    # layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    # layers.MaxPool2D(),

    # Head
    layers.BatchNormalization(renorm=True),
    layers.Flatten(),
    #layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

optimizer = tf.keras.optimizers.Adam(epsilon=0.01)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

history = model.fit(
    ds_train,
    validation_data=ds_train , #remember to change this later
    epochs=5,
)

# Plot learning curves

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
plt.show()

