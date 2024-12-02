import math
import numpy as np
import pandas as pd

import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import keras.utils
import pathlib


def build_model(bottom_model, classes):
    model = bottom_model.layers[-2].output
    model = GlobalAveragePooling2D()(model)
    model = Dense(classes, activation="softmax", name="out_layer")(model)

    return model


input = pathlib.Path(__file__).parents[1] / "data" / "fer2013" / "fer2013.csv"


df = pd.read_csv(input)

img_array = df.pixels.apply(
    lambda x: np.array(x.split(" ")).reshape(48, 48).astype("float32")
)
img_array = np.stack(img_array, axis=0)

img_features = []

for i in range(len(img_array)):
    temp = cv2.cvtColor(img_array[i], cv2.COLOR_GRAY2RGB)
    img_features.append(temp)

img_features = np.array(img_features)
le = LabelEncoder()

img_labels = le.fit_transform(df.emotion)
img_labels = keras.utils.to_categorical(img_labels)
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
X_train, X_valid, y_train, y_valid = train_test_split(
    img_features,
    img_labels,
    shuffle=True,
    stratify=img_labels,
    test_size=0.1,
    random_state=42,
)

img_width = X_train.shape[1]
img_height = X_train.shape[2]
img_depth = X_train.shape[3]
num_classes = y_train.shape[1]

X_train = X_train / 255
X_valid = X_valid / 255

vgg = tf.keras.applications.VGG19(
    weights="imagenet", include_top=False, input_shape=(48, 48, 3)
)

head = build_model(vgg, num_classes)

model = Model(inputs=vgg.input, outputs=head, use_multiprocessing=True)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    min_delta=0.00005,
    patience=11,
    verbose=1,
    restore_best_weights=True,
)

lr_scheduler = ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    verbose=1,
)

callbacks = [
    early_stopping,
    lr_scheduler,
]

batch_size = 32
epochs = 25
optims = [
    optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999),
]

model.compile(
    loss="categorical_crossentropy", optimizer=optims[0], metrics=["accuracy"]
)

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
)
train_datagen.fit(X_train)

# batch size of 32 performs the best.
batch_size = 32
epochs = 25
optims = [
    optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999),
]

model.compile(
    loss="categorical_crossentropy", optimizer=optims[0], metrics=["accuracy"]
)

history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_valid, y_valid),
    steps_per_epoch=round(len(X_train) / batch_size),
    epochs=epochs,
    callbacks=callbacks,
)

model_yaml = model.to_json()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)

model.save("model.h5")
