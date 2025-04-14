import os

import tensorflow.keras
import numpy as np
from PIL import Image
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
import tensorflow as tf
from tqdm import tqdm

from tensorflow.keras import backend as K


def read_image(image_path, target_size=(64, 64)):
    image = Image.open(image_path)
    grayscale_image = image.convert('L')
    resized_image = grayscale_image.resize(target_size)
    image_array = np.array(resized_image)
    image_1d_array = image_array.flatten()

    return image_1d_array


def read_data(dict_path, target_size=(64, 64)):
    classes = os.listdir(dict_path)

    X = []
    y = []

    class_id = 0

    for class_instance in classes:
        emotion_path = os.path.join(dict_path, class_instance)
        print(class_instance)

        list_dir = os.listdir(emotion_path)

        for img_idx in tqdm(range(len(list_dir))):
            image_name = list_dir[img_idx]
            image_path = os.path.join(emotion_path, image_name)
            image = read_image(image_path, target_size=target_size)

            X.append(image)
            y.append(class_id)

        class_id += 1

    X = np.array(X)
    y = np.array(y)

    indices = np.random.choice(len(y), size=len(y), replace=False)

    return X[indices], y[indices]


def save_data_to_npz(data_path, save_path, target_size=(64, 64)):
    X, y = read_data(data_path, target_size=target_size)
    np.savez(save_path, X=X, y=y)


def create_data_npz(dict_path, target_size=(64, 64)):
    data_path = os.path.join(dict_path, "data.npz")

    if not os.path.exists(data_path):
        save_data_to_npz(os.path.join(dict_path, "images"), data_path, target_size=target_size)


def read_data_npz(dict_path):
    read_data = np.load(dict_path)
    X_loaded = read_data['X']
    y_loaded = read_data['y']

    return X_loaded, y_loaded


def train_model():
    if os.path.exists('data\\model.h5'):
        return tf.keras.models.load_model('data\\model.h5')

    neural_net = Sequential()

    neural_net.add(Conv2D(32, activation='relu', kernel_size=(3, 3), input_shape=(128, 128, 1)))
    neural_net.add(MaxPooling2D(2, 2))
    neural_net.add(Conv2D(64, activation='relu', kernel_size=(3, 3)))
    neural_net.add(MaxPooling2D(2, 2))
    neural_net.add(Conv2D(128, activation='relu', kernel_size=(3, 3)))
    neural_net.add(MaxPooling2D(2, 2))

    neural_net.add(Flatten())

    neural_net.add(Dense(1024, activation='relu'))
    neural_net.add(Dropout(0.2))
    neural_net.add(Dense(512, activation='relu'))
    neural_net.add(Dropout(0.2))
    neural_net.add(Dense(128, activation='relu'))
    neural_net.add(Dropout(0.1))

    neural_net.add(Dense(2, activation='sigmoid'))

    neural_net.compile(optimizer=Adam(learning_rate=0.00001, decay=1e-6), loss='categorical_crossentropy',
                       metrics=['accuracy'])

    print(y_train.shape)
    K.clear_session()

    neural_net.fit(X_train, y_train, epochs=25)
    neural_net.save('data\\model.h5')

    return neural_net


def test_model():
    predictions = tf.argmax(nn.predict(X_test), axis=1)
    right_guesses = 0
    tst = tf.argmax(y_test, axis=1)

    for idx in range(len(predictions)):
        if predictions[idx] == tst[idx]:
            right_guesses += 1

    return right_guesses / len(predictions)


create_data_npz('data', target_size=(128, 128))

X, y = read_data_npz('data\\data.npz')

X = tf.reshape(X, (len(X), 128, 128, 1))
y = tf.reshape(y, (len(X), 1))
y = to_categorical(y, num_classes=2)

split_ratio = 0.8

num_samples = len(X)

indices = np.arange(num_samples)
np.random.shuffle(indices)

indices = tf.convert_to_tensor(indices, dtype=tf.int32)

split_index = int(num_samples * split_ratio)

train_indices = indices[:split_index]
test_indices = indices[split_index:]

X_train = tf.gather(X, train_indices)
X_test = tf.gather(X, test_indices)
y_train = tf.gather(y, train_indices)
y_test = tf.gather(y, test_indices)

nn = train_model()
print(test_model())

# img = read_image('data\\images\\test_images\\img.png', (128, 128))
# img = tf.reshape(img, (1 , 128, 128, 1))
# print('cat' if tf.argmax(nn.predict(img)[0]) == 0 else 'dog')
