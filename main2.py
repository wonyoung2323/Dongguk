import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import numpy as np
# from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import CSVLogger
from prune import inceptionv3

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import tensorflow as tf
from tensorflow.python.client import device_lib
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print(get_available_devices())


# for restrict gpu capacity
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# dimensions of the images.
img_width, img_height = 299, 299
output_dir = '/home/finalproject1/PycharmProjects/pythonProject2/prune/'
train_data_dir = '/home/finalproject1/Downloads/train'
validation_data_dir = '/home/finalproject1/Downloads/test'
top_model_weights_path = output_dir+'top_model_weights.h5'
tuned_weights_path = output_dir+'tuned_weights.h5'
nb_train_samples = 3170
nb_validation_samples = 500
top_epochs = 20
tune_epochs = 5
batch_size = 8


# def save_bottleneck_features():
#     # build the Inception V3 network
#     model = inceptionv3.InceptionV3(include_top=False,
#                                     weights='imagenet',
#                                     input_tensor=None,
#                                     input_shape=None,
#                                     pooling='avg')
#
#     # Save the bottleneck features for the training data set
#     datagen = ImageDataGenerator(
#         preprocessing_function=inceptionv3.preprocess_input)
#     train_data = datagen.flow_from_directory(train_data_dir,
#                                             target_size=(img_width, img_height),
#                                             batch_size=batch_size,
#                                             class_mode='sparse',
#                                             shuffle=False)
#     features = model.predict(train_data)
#     labels = np.eye(train_data.num_classes, dtype='uint8')[train_data.classes]
#     np.save(output_dir+'bottleneck_features_train.npy', features)
#     np.save(output_dir+'bottleneck_labels_train.npy', labels)
#
#     # Save the bottleneck features for the validation data set
#     val_data = datagen.flow_from_directory(validation_data_dir,
#                                             target_size=(img_width, img_height),
#                                             batch_size=batch_size,
#                                             class_mode=None,
#                                             shuffle=False)
#     features = model.predict(val_data)
#     labels = np.eye(val_data.num_classes, dtype='uint8')[val_data.classes]
#     np.save(output_dir+'bottleneck_features_validation.npy', features)
#     np.save(output_dir+'bottleneck_labels_validation.npy', labels)


def train_top_model():
    # Load the bottleneck features and labels
    train_features = np.load(output_dir+'bottleneck_features_train.npy')
    train_labels = np.load(output_dir+'bottleneck_labels_train.npy')
    validation_features = np.load(output_dir+'bottleneck_features_validation.npy')
    validation_labels = np.load(output_dir+'bottleneck_labels_validation.npy')

    # Create the top model for the inception V3 network, a single Dense layer
    # with softmax activation.
    top_input = Input(shape=train_features.shape[1:])
    top_output = Dense(18, activation='softmax')(top_input)
    model = Model(top_input, top_output)

    # Train the model using the bottleneck features and save the weights.
    model.compile(optimizer=SGD(lr=1e-4, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    csv_logger = CSVLogger(output_dir + 'top_model_training.csv')
    model.fit(train_features, train_labels,
              epochs=top_epochs,
              batch_size=batch_size,
              validation_data=(validation_features, validation_labels),
              callbacks=[csv_logger])
    model.save_weights(top_model_weights_path)


def tune_model():
    # Build the Inception V3 network.
    base_model = inceptionv3.InceptionV3(include_top=False,
                                          weights='imagenet',
                                          pooling='avg')
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    top_input = Input(shape=base_model.output_shape[1:])
    top_output = Dense(18, activation='softmax')(top_input)
    top_model = Model(top_input, top_output)

    # Note that it is necessary to start with a fully-trained classifier,
    # including the top classifier, in order to successfully do fine-tuning.
    top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    model = Model(inputs=base_model.inputs,
                  outputs=top_model(base_model.outputs))

    # Set all layers up to 'mixed8' to non-trainable (weights will not be updated)
    last_train_layer = model.get_layer(name='mixed8')
    for layer in model.layers[:model.layers.index(last_train_layer)]:
        layer.trainable = False

    # Compile the model with a SGD/momentum optimizer and a very slow learning rate.
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # Prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        preprocessing_function=inceptionv3.preprocess_input,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(
        preprocessing_function=inceptionv3.preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    loss = model.evaluate(validation_generator)
    print('Model validation performance before fine-tuning:', loss)

    csv_logger = CSVLogger(output_dir+'model_tuning.csv')
    # fine-tune the model
    model.fit(train_generator,
              epochs=tune_epochs,
              validation_data=validation_generator,
              workers=4,
              callbacks=[csv_logger])
    model.save(tuned_weights_path)


if __name__ == '__main__':
    #save_bottleneck_features()
    train_top_model()
    tune_model()