import logging
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2


cap  = cv2.VideoCapture(0)

my720x1080Image = tf.random.uniform(
    shape=(720, 1080),
    minval=0,
    maxval=1.0,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)
print(my720x1080Image.shape)


# model = tf.keras.Sequential()
# model.add()

plt.imshow(my720x1080Image, cmap="gray", interpolation="bicubic")
plt.show()