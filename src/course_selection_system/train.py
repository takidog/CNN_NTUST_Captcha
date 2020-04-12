import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

epochs = 20
batch_size = 100
img_h = 13
img_w = 57
num_length = 6
x_list = []
y_list = []
x_train = []
y_train = []
x_test = []
y_test = []
char_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17,
             'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35}


def to_onehot(text):

    size = len(char_dict.keys())
    onehot = [0 for _ in range(size)]
    onehot[char_dict[text]] = 1

    return onehot


def split_char_image(img_filename, img_array):
    for i in range(num_length):
        step = img_w // num_length
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255)
        y_list.append(img_filename[i])


img_filenames = os.listdir('dataset')

for img_filename in img_filenames:
    if '.jpg' not in img_filename:
        continue
    img = load_img('dataset/{0}'.format(img_filename),
                   color_mode='grayscale')
    img_array = img_to_array(img)
    img_rows, img_cols, _ = img_array.shape
    split_char_image(img_filename, img_array)

y_list = [to_onehot(i) for i in y_list]

x_train, x_test, y_train, y_test = train_test_split(x_list, y_list)

if os.path.isfile('cnn_model.h5'):
    model = models.load_model('cnn_model.h5')
    print('Model loaded from file.')
else:
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                            input_shape=(img_h, img_w // num_length, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(rate=0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(36, activation='softmax'))
    print('New model created.')
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

model.fit(np.array(x_train), np.array(y_train), batch_size=300,
          epochs=epochs, verbose=1, validation_data=(np.array(x_test), np.array(y_test)))

loss, accuracy = model.evaluate(np.array(x_test), np.array(y_test), verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

model.save('cnn_model.h5')
