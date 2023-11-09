import math
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

def build_model(bottom_model, classes):
    model = bottom_model.layers[-2].output
    model = GlobalAveragePooling2D()(model)
    model = Dense(classes, activation = 'softmax', name = 'out_layer')(model)

    return model

def modeling(img_features,img_labels, batch_size = 32, epochs = 25):
    """
    input: img_features,img_labels
    img_features: numpy array of all the images read. Hence the input will be the numpy images
    img_labels: numpy array of the target variables.
    Thus,
    img_features = X
    img_labels = y

    output: performane table sorted with the accuracies

    batch_size and epochs are hyper parameters. Change them accordingly to the requirement.
    """
    X_train, X_valid, y_train, y_valid = train_test_split(img_features,
                                                          img_labels,
                                                          shuffle = True,
                                                          stratify = img_labels,
                                                          test_size = 0.1,
                                                          random_state = 42)
    X_train.shape, X_valid.shape, y_train.shape, y_valid.shape

    num_classes = y_train.shape[1]

    X_train = X_train / 255.
    X_valid = X_valid / 255.

    """# VGG19 Network"""

    vgg = tf.keras.applications.VGG19(weights = 'imagenet',
                                      include_top = False,
                                      input_shape = (48, 48, 3))

    head = build_model(vgg, num_classes)
    model = Model(inputs = vgg.input, outputs = head)
    early_stopping = EarlyStopping(monitor = 'val_accuracy',
                                   min_delta = 0.00005,
                                   patience = 11,
                                   verbose = 1,
                                   restore_best_weights = True,)
    lr_scheduler = ReduceLROnPlateau(monitor = 'val_accuracy',
                                     factor = 0.5,
                                     patience = 7,
                                     min_lr = 1e-7,
                                     verbose = 1,)

    callbacks = [early_stopping,lr_scheduler,]

    train_datagen = ImageDataGenerator(rotation_range = 15,
                                       width_shift_range = 0.15,
                                       height_shift_range = 0.15,
                                       shear_range = 0.15,
                                       zoom_range = 0.15,
                                       horizontal_flip = True,)
    train_datagen.fit(X_train)

    optims = [optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.9, beta_2 = 0.999),]

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optims[0],
                  metrics = ['accuracy'])

    history = model.fit(train_datagen.flow(X_train,
                                           y_train,
                                           batch_size = batch_size),
                                           validation_data = (X_valid, y_valid),
                                           steps_per_epoch = len(X_train) / batch_size,
                                           epochs = epochs,
                                           callbacks = callbacks,
                                           use_multiprocessing = True)

    yhat_valid_vgg = np.argmax(model.predict(X_valid), axis=1)

    """# ResNet 50"""

    res = tf.keras.applications.ResNet50(weights = 'imagenet',
                                      include_top = False,
                                      input_shape = (48, 48, 3))

    head = build_model(res, num_classes)

    model = Model(inputs = res.input, outputs = head)

    early_stopping = EarlyStopping(monitor = 'val_accuracy',
                                   min_delta = 0.00005,
                                   patience = 11,
                                   verbose = 1,
                                   restore_best_weights = True,)

    lr_scheduler = ReduceLROnPlateau(monitor = 'val_accuracy',
                                     factor = 0.5,
                                     patience = 7,
                                     min_lr = 1e-7,
                                     verbose = 1,)

    callbacks = [early_stopping,lr_scheduler,]

    train_datagen = ImageDataGenerator(rotation_range = 15,
                                       width_shift_range = 0.15,
                                       height_shift_range = 0.15,
                                       shear_range = 0.15,
                                       zoom_range = 0.15,
                                       horizontal_flip = True,)
    train_datagen.fit(X_train)

    optims = [optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.9, beta_2 = 0.999),]

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optims[0],
                  metrics = ['accuracy'])

    history = model.fit(train_datagen.flow(X_train,
                                           y_train,
                                           batch_size = batch_size),
                                           validation_data = (X_valid, y_valid),
                                           steps_per_epoch = len(X_train) / batch_size,
                                           epochs = epochs,
                                           callbacks = callbacks,
                                           use_multiprocessing = True)

    yhat_valid_res = np.argmax(model.predict(X_valid), axis=1)

    """# MobileNet"""

    res = tf.keras.applications.MobileNet(weights = 'imagenet',
                                      include_top = False,
                                      input_shape = (48, 48, 3))

    head = build_model(res, num_classes)
    model = Model(inputs = res.input, outputs = head)

    early_stopping = EarlyStopping(monitor = 'val_accuracy',
                                   min_delta = 0.00005,
                                   patience = 11,
                                   verbose = 1,
                                   restore_best_weights = True,)

    lr_scheduler = ReduceLROnPlateau(monitor = 'val_accuracy',
                                     factor = 0.5,
                                     patience = 7,
                                     min_lr = 1e-7,
                                     verbose = 1,)

    callbacks = [early_stopping,lr_scheduler]

    train_datagen = ImageDataGenerator(rotation_range = 15,
                                       width_shift_range = 0.15,
                                       height_shift_range = 0.15,
                                       shear_range = 0.15,
                                       zoom_range = 0.15,
                                       horizontal_flip = True,)
    train_datagen.fit(X_train)

    optims = [optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.9, beta_2 = 0.999),]

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optims[0],
                  metrics = ['accuracy'])

    history = model.fit(train_datagen.flow(X_train,
                                           y_train,
                                           batch_size = batch_size),
                                           validation_data = (X_valid, y_valid),
                                           steps_per_epoch = len(X_train) / batch_size,
                                           epochs = epochs,
                                           callbacks = callbacks,
                                           use_multiprocessing = True)

    yhat_valid_mob = np.argmax(model.predict(X_valid), axis=1)

    """# EfficientNetB1"""

    res = tf.keras.applications.EfficientNetB7(weights = 'imagenet',
                                      include_top = False,
                                      input_shape = (48, 48, 3))

    head = build_model(res, num_classes)
    model = Model(inputs = res.input, outputs = head)
    early_stopping = EarlyStopping(monitor = 'val_accuracy',
                                   min_delta = 0.00005,
                                   patience = 11,
                                   verbose = 1,
                                   restore_best_weights = True,)

    lr_scheduler = ReduceLROnPlateau(monitor = 'val_accuracy',
                                     factor = 0.5,
                                     patience = 7,
                                     min_lr = 1e-7,
                                     verbose = 1,)

    callbacks = [early_stopping,lr_scheduler,]

    train_datagen = ImageDataGenerator(rotation_range = 15,
                                       width_shift_range = 0.15,
                                       height_shift_range = 0.15,
                                       shear_range = 0.15,
                                       zoom_range = 0.15,
                                       horizontal_flip = True,)
    train_datagen.fit(X_train)

    optims = [optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.9, beta_2 = 0.999),]

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optims[0],
                  metrics = ['accuracy'])

    history = model.fit(train_datagen.flow(X_train,
                                           y_train,
                                           batch_size = batch_size),
                                           validation_data = (X_valid, y_valid),
                                           steps_per_epoch = len(X_train) / batch_size,
                                           epochs = epochs,
                                           callbacks = callbacks,
                                           use_multiprocessing = True)

    yhat_valid_eff = np.argmax(model.predict(X_valid), axis=1)

    """# DenseNet"""

    res = tf.keras.applications.DenseNet121(weights = 'imagenet',
                                      include_top = False,
                                      input_shape = (48, 48, 3))

    head = build_model(res, num_classes)

    model = Model(inputs = res.input, outputs = head)

    early_stopping = EarlyStopping(monitor = 'val_accuracy',
                                   min_delta = 0.00005,
                                   patience = 11,
                                   verbose = 1,
                                   restore_best_weights = True,)

    lr_scheduler = ReduceLROnPlateau(monitor = 'val_accuracy',
                                     factor = 0.5,
                                     patience = 7,
                                     min_lr = 1e-7,
                                     verbose = 1,)

    callbacks = [early_stopping,lr_scheduler,]

    train_datagen = ImageDataGenerator(rotation_range = 15,
                                       width_shift_range = 0.15,
                                       height_shift_range = 0.15,
                                       shear_range = 0.15,
                                       zoom_range = 0.15,
                                       horizontal_flip = True,)
    train_datagen.fit(X_train)

    optims = [optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.9, beta_2 = 0.999),]

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optims[0],
                  metrics = ['accuracy'])

    history = model.fit(train_datagen.flow(X_train,
                                           y_train,
                                           batch_size = batch_size),
                                           validation_data = (X_valid, y_valid),
                                           steps_per_epoch = len(X_train) / batch_size,
                                           epochs = epochs,
                                           callbacks = callbacks,
                                           use_multiprocessing = True)

    yhat_valid_den = np.argmax(model.predict(X_valid), axis=1)

    y_true = np.argmax(y_valid, axis=1)
    accuracy_vgg = accuracy_score(y_true, yhat_valid_vgg)
    precision_vgg = precision_score(y_true, yhat_valid_vgg, average='weighted')
    recall_vgg = recall_score(y_true, yhat_valid_vgg, average='weighted')
    f1_vgg = f1_score(y_true, yhat_valid_vgg, average='weighted')

    accuracy_res = accuracy_score(y_true, yhat_valid_res)
    precision_res = precision_score(y_true, yhat_valid_res, average='weighted')
    recall_res = recall_score(y_true, yhat_valid_res, average='weighted')
    f1_res = f1_score(y_true, yhat_valid_res, average='weighted')

    accuracy_mob = accuracy_score(y_true, yhat_valid_mob)
    precision_mob = precision_score(y_true, yhat_valid_mob, average='weighted')
    recall_mob = recall_score(y_true, yhat_valid_mob, average='weighted')
    f1_mob = f1_score(y_true, yhat_valid_mob, average='weighted')

    accuracy_eff = accuracy_score(y_true, yhat_valid_eff)
    precision_eff = precision_score(y_true, yhat_valid_eff, average='weighted')
    recall_eff = recall_score(y_true, yhat_valid_eff, average='weighted')
    f1_eff = f1_score(y_true, yhat_valid_eff, average='weighted')

    accuracy_den = accuracy_score(y_true, yhat_valid_den)
    precision_den = precision_score(y_true, yhat_valid_den, average='weighted')
    recall_den = recall_score(y_true, yhat_valid_den, average='weighted')
    f1_den = f1_score(y_true, yhat_valid_den, average='weighted')


    data = {
    'Model': ['VGGNet', 'ResNet', 'MobileNet', 'EfficientNet', 'DenseNet'],
    'Accuracy': [accuracy_vgg, accuracy_res, accuracy_mob, accuracy_eff, accuracy_den],
    'Precision': [precision_vgg, precision_res, precision_mob, precision_eff, precision_den],
    'Recall': [recall_vgg, recall_res, recall_mob, recall_eff, recall_den],
    'F1 Score': [f1_vgg, f1_res, f1_mob, f1_eff, f1_den]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Set 'Model' as the index
    df.set_index('Model', inplace=True)

    df = df.sort_values(by='Accuracy', ascending=False)
    return df