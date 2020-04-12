import numpy as np
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import requests
from PIL import Image
import os
num_length = 6


def split_char_image(img_array, img_w):
    x_list = []
    for i in range(num_length):
        step = img_w // num_length
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255)
    return x_list


char_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17,
             'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35}
reverse_dict = {}
for k, v in char_dict.items():
    reverse_dict.update({str(v): k})

model = models.load_model('cnn_model.h5')

img_filenames = os.listdir('verify')

for img_filename in img_filenames:
    # test 10 times.

    original = Image.open(
        'verify/{filename}'.format(
            filename=img_filename)).convert('L')

    # w:61 h:14
    original = original.crop((10, 10, 124, 36))
    original = original.resize((57, 13))

    img_array = img_to_array(original)
    img_h, img_w, _ = img_array.shape
    x_list = split_char_image(img_array, img_w)

    varification_code = []
    for i in range(num_length):
        confidences = model.predict(np.array([x_list[i]]), verbose=0)
        result_class = model.predict_classes(np.array([x_list[i]]), verbose=0)
        varification_code.append(result_class[0])

    code = "".join([reverse_dict[str(i)] for i in varification_code])
    print(img_filename)
    print(code)
