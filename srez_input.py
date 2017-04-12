import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def setup_inputs(sess, filenames, image_size=None, capacity_factor=3):

    if image_size is None:
        image_size = FLAGS.sample_size
    
    # Read each JPEG file
    reader = tf.WholeFileReader()
    filename_queue = tf.train.string_input_producer(filenames)
    key, value = reader.read(filename_queue)
    channels = FLAGS.channels
    image = tf.image.decode_jpeg(value, channels=channels, name="dataset_image")
    image.set_shape([None, None, channels])

    # Crop and other random augmentations
    if FLAGS.reflect_images:
        image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, .95, 1.05)
    image = tf.image.random_brightness(image, .05)
    image = tf.image.random_contrast(image, .95, 1.05)

    wiggle = 8
    off_x, off_y = 25-wiggle, 60-wiggle
    crop_size = 128
    crop_size_plus = crop_size + 2*wiggle
    
    # Upsizes mnist images since they're too small
    if FLAGS.mnist:
        image = tf.reshape(image, (1, 28, 28, channels))    
        image = tf.image.resize_nearest_neighbor(image, (crop_size_plus, crop_size_plus))
        image = tf.reshape(image, (crop_size_plus, crop_size_plus, channels))
    
    else:
        image = tf.image.crop_to_bounding_box(image, off_y, off_x, crop_size_plus, crop_size_plus)

    image = tf.random_crop(image, [crop_size, crop_size, channels])
    image = tf.reshape(image, [1, crop_size, crop_size, channels])
    image = tf.cast(image, tf.float32)/255.0

    if crop_size != image_size:
        image = tf.image.resize_area(image, [image_size, image_size])

    # The feature is simply a Kx downscaled version
    K = FLAGS.downscale
    downsampled = tf.image.resize_area(image, [image_size//K, image_size//K])

    feature = tf.reshape(downsampled, [image_size//K, image_size//K, channels])
    label   = tf.reshape(image,       [image_size,   image_size,     channels])

    # Using asynchronous queues
    features, labels = tf.train.batch([feature, label],
                                      batch_size=FLAGS.batch_size,
                                      num_threads=4,
                                      capacity = capacity_factor*FLAGS.batch_size,
                                      name='labels_and_features')

    tf.train.start_queue_runners(sess=sess)
      
    return features, labels
