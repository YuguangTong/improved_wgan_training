import tensorflow as tf
import random
import os
from scipy.misc import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

channels = 3
image_size = 64
# The feature is simply a Kx downscaled version
# K = 4
# image_size_k = image_size // K

input_dir = 'dataset'
output_dir = 'celebA_64x64'


with tf.Session() as sess:
    inp = tf.placeholder(tf.uint8, [None, None, channels])
    image = tf.image.random_flip_left_right(inp)
    image = tf.image.random_saturation(image, .95, 1.05)
    image = tf.image.random_brightness(image, .05)
    image = tf.image.random_contrast(image, .95, 1.05)

    wiggle = 8
    off_x, off_y = 25-wiggle, 60-wiggle
    crop_size = 128
    crop_size_plus = crop_size + 2*wiggle
    image = tf.image.crop_to_bounding_box(image, off_y, off_x, crop_size_plus, crop_size_plus)
    image = tf.random_crop(image, [crop_size, crop_size, 3])

    image = tf.reshape(image, [1, crop_size, crop_size, 3])
    if crop_size != image_size:
        image = tf.image.resize_area(image, [image_size, image_size])

    image_resize = tf.reshape(image, [image_size, image_size, 3])
    image_resize = tf.cast(image_resize, tf.uint8)
    # read files
    input_files = tf.gfile.ListDirectory(input_dir)
    input_files = sorted(input_files)
    random.shuffle(input_files)

    for f in tqdm(input_files):
        input_file = os.path.join(input_dir, f)
        output_file = os.path.join(output_dir, f)
        input_image = imread(input_file)
        output_image = sess.run(image_resize, feed_dict={inp: input_image})
        imsave(output_file, output_image)
