import tensorflow as tf
import numpy as np
import sys
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os


model_path = "model.tflite"
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
num_length = 6

image_h = 13
image_w = 57

char_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17,
             'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35}
reverse_dict = {}
for k, v in char_dict.items():
    reverse_dict.update({str(v): k})


def split_char_image(img_array):
    x_list = []
    for i in range(num_length):
        step = image_w // num_length
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255)
    return x_list


if __name__ == "__main__":

    img_filenames = os.listdir('verify')

    for img_filename in img_filenames:
        original = Image.open('verify/{}'.format(img_filename)
                              ).convert('L')  # convert grayscale
        # w:61 h:14
        original = original.crop((10, 10, 124, 36))
        original = original.resize((57, 13))

        img_array = img_to_array(original)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        x_list = split_char_image(img_array)
        # print(np.array(x_list).shape)
        # (6,14,10,1)
        res = []

        # input need (1,14,10,1)
        for i in x_list:
            interpreter.set_tensor(input_details[0]['index'], [i])
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            results = np.squeeze(output_data)
            top = results.argsort()[-1:]
            res.append(reverse_dict[str(top[0])])
        print(img_filename)
        print("".join([i for i in res]))
