import os
from flatbuffers.builder import np
import inception_v3

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import tensorflow as tf
import tensorflow.keras as keras

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from tensorflow.python.client import device_lib
from keras.preprocessing.image import img_to_array, load_img
import glob
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools

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


IMAGE_SIZE = [224, 224]

train_path = '/home/finalproject1/Downloads/train'
valid_path = '/home/finalproject1/Downloads/test'

tf.keras.applications.InceptionV3(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

inception = inception_v3.InceptionV3(input_shape=IMAGE_SIZE + [3], weights ='imagenet', include_top=False, )

for layer in inception.layers:
    layer.trainable = False

folders = glob.glob('/home/finalproject1/Downloads/train/*')
folders


from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

x = Flatten()(inception.output)

#prediction = Dense(len(folders), activation='softmax', kernel_regularizer=regularizers.l2(0.0001))(x)
prediction = Dense(len(folders), activation='softmax')(x)

model = keras.Model(inputs=inception.input, outputs=prediction)

model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics=['accuracy']
)
import random

def add_noise(img):
    '''Add random noise to an image'''
    VARIABILITY = 50
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        if input_img.ndim == 3:
            img_h, img_w, img_c = input_img.shape
        elif input_img.ndim == 2:
            img_h, img_w = input_img.shape

        p_1 = random.randrange(1,11)

        if p_1 > 3:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r) / 2)
            h = int(np.sqrt(s * r) / 2)
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            if input_img.ndim == 3:
                c = np.random.uniform(v_l, v_h, (h, w, img_c))
            if input_img.ndim == 2:
                c = np.random.uniform(v_l, v_h, (h, w))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w] = c

        return input_img

    return eraser

# Prepare data-augmenting data generator
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   # preprocessing_function=add_noise,
                                   preprocessing_function=get_random_eraser(v_l=0, v_h=1))

img = load_img('/home/finalproject1/PycharmProjects/pythonProject2/2.jpg')  # PIL 이미지
x = img_to_array(img)  # (3, 150, 150) 크기의 NumPy 배열
x = x.reshape((1,) + x.shape)  # (1, 3, 150, 150) 크기의 NumPy 배열

# 아래 .flow() 함수는 임의 변환된 이미지를 배치 단위로 생성해서
# 지정된 `preview/` 폴더에 저장합니다.
i = 0
for batch in train_datagen.flow(x, batch_size=1,
                          save_to_dir='/home/finalproject1/PycharmProjects/pythonProject2/picture', save_prefix='b', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # 이미지 20장을 생성하고 마칩니다

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size=(224, 224),
                                                 batch_size=8,
                                                 class_mode='categorical', shuffle=True)

test_set = test_datagen.flow_from_directory(valid_path,
                                            target_size = (224, 224),
                                            batch_size =8,
                                            class_mode = 'categorical', shuffle=False)

valid_datagen = ImageDataGenerator(rescale = 1./255)

valid_generator = valid_datagen.flow_from_directory(valid_path,
                                                    target_size=(224, 224),
                                                    batch_size=8,
                                                    class_mode='categorical', shuffle=True)

r = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=30,
    workers=4,     # to generate data with multi-cpu core
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)


print("-- Evaluate --")
scores = model.evaluate_generator(test_set, steps=len(test_set))
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# save it as a h5 file

model.save('./model_resnet50.h5')
saved_model_dir = '/home/finalproject1/PycharmProjects/pythonProject2'
tf.saved_model.save(model, saved_model_dir)
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
   f.write(tflite_model)

#labels = '\n'.join(sorted(training_set.class_indices.keys()))
#with open('labels.txt', 'w') as f:
#    f.write(labels)

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('./') # path to the SavedModel directory
tflite_model = converter.convert()
# Save the model.
with open('model.tflite', 'wb') as f:
 f.write(tflite_model)

#Print the Target names
target_names = []

for key in training_set.class_indices:

    target_names.append(key)


# print(target_names)

def plot_confusion_matrix(cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(11, 11))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 2

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

Y_pred = model.predict_generator(test_set)

y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')

cm = confusion_matrix(test_set.classes, y_pred)

plot_confusion_matrix(cm, target_names, title='Confusion Matrix')



#Print Classification Report

print('Classification Report')

print(classification_report(test_set.classes, y_pred, target_names=target_names))

plt.show()