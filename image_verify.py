# Will raise an error if any image in the directory is not a valid .jpg

import tensorflow as tf
import glob
import os

for i, image_name in enumerate(glob.glob(os.path.join(os.getcwd(), '*.jpg'))):
    print i, image_name
    with tf.Graph().as_default():
        image_contents = tf.read_file(image_name)
        image = tf.image.decode_jpeg(image_contents, channels=3)
        init_op = tf.initialize_all_tables()
        with tf.Session() as sess:
            sess.run(init_op)
            tmp = sess.run(image)
