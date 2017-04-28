import srez_demo
import srez_input
import srez_model
import srez_train

import os.path
import random
import numpy as np
import numpy.random

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Configuration (alphabetically)

tf.app.flags.DEFINE_integer('batch_size', 8,
                            "Number of samples per batch.")

tf.app.flags.DEFINE_integer('channels', 3,
                            "Number of color channels to use. [1|3]")

tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint',
                           "Output folder where checkpoints are dumped.")

tf.app.flags.DEFINE_integer('checkpoint_period', 5000,
                            "Number of batches in between checkpoints")

tf.app.flags.DEFINE_string('dataset', 'dataset',
                           "Path to the train dataset directory.")

tf.app.flags.DEFINE_integer('downscale', 12,
                            "How much downscaling should be performed on the input images")

tf.app.flags.DEFINE_float('epsilon', 1e-1,
                          "Fuzz term to avoid numerical instability")

tf.app.flags.DEFINE_string('run', 'train',
                            "Which operation to run. [demo|train]")

tf.app.flags.DEFINE_float('gene_l1_factor', 0.9999,
                          "Multiplier for generator L1 loss term (essentially, how important is generator L1 loss compared to cross entropy loss")

tf.app.flags.DEFINE_float('disc_real_factor', 0.6,
                          "Mutliplier for how much the real_D loss is weighted compared to the fake_D loss (0.5 is equal)")

tf.app.flags.DEFINE_float('gpu_fraction', 0.75,
                          "Percent of GPU memory to be used")

tf.app.flags.DEFINE_float('learning_beta1_G', 0.5,
                          "Beta1 parameter used for AdamOptimizer (Generator)")

tf.app.flags.DEFINE_float('learning_beta1_D', 0.9,
                          "Beta1 parameter used for AdamOptimizer (Discriminator)")

tf.app.flags.DEFINE_float('learning_rate_start', 0.0010,
                          "Starting learning rate used for AdamOptimizer")

tf.app.flags.DEFINE_integer('learning_rate_half_life', 5000,
                            "Number of batches until learning rate is halved")

tf.app.flags.DEFINE_bool('log_device_placement', False,
                         "Log the device where variables are placed.")

tf.app.flags.DEFINE_bool('upsize', False,
                         "Set True for MNIST images or other small images that require upsizing.")

tf.app.flags.DEFINE_bool('reflect_images', False,
                         "Whether to reflect the images during training")

tf.app.flags.DEFINE_bool('specific_test', True,
                         "Set to True if you want to test specific images (in the test folder)")                         

tf.app.flags.DEFINE_integer('sample_size', 224,
                            "Image size in pixels.")

tf.app.flags.DEFINE_integer('summary_period', 100,
                            "Number of batches between summary data dumps")

tf.app.flags.DEFINE_integer('random_seed', 0,
                            "Seed used to initialize rng.")

tf.app.flags.DEFINE_string('test_dataset', 'test',
                           "Path to the test dataset directory (optional).")

tf.app.flags.DEFINE_integer('test_vectors', 16,
                            "Number of images from the training set to set aside for testing")
                            
tf.app.flags.DEFINE_string('train_dir', 'train',
                           "Output folder where training logs are dumped.")

tf.app.flags.DEFINE_integer('train_time', 30,
                            "Time in minutes to train the model")

tf.app.flags.DEFINE_float('disc_learning_rate_multiplier', 0.05,
                            "Adjusts learning rate of the discriminator compared to the generator")

tf.app.flags.DEFINE_string('comparison_image', 'original',
                            "Whether to compare the GAN image against original or downscaled image. [original|downscaled]")

tf.app.flags.DEFINE_bool('use_L2_norm', True,
                            "Replace the L1 norm with the L2 norm")

tf.app.flags.DEFINE_bool('swap_discriminator', True,
                            "Swaps the real and fake discriminator losses")


def prepare_dirs(delete_train_dir=False):
    # Create checkpoint dir (do not delete anything)
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    
    # Cleanup train dir
    # ----
    # This was removed to ensure Windows compatiblity
    # ----
    #if delete_train_dir:
    #    if tf.gfile.Exists(FLAGS.train_dir):
    #        tf.gfile.Remove(FLAGS.train_dir)
    #    tf.gfile.MakeDirs(FLAGS.train_dir)

    # Return names of training files
    if not tf.gfile.Exists(FLAGS.dataset) or \
       not tf.gfile.IsDirectory(FLAGS.dataset):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.dataset,))

    filenames = tf.gfile.ListDirectory(FLAGS.dataset)
    random.shuffle(filenames)
    filenames = [os.path.join(FLAGS.dataset, f) for f in filenames]

    return filenames

def prepare_test_dirs():
    # Create checkpoint dir (do not delete anything)
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    
    # Cleanup train dir
    # ----
    # This was removed to ensure Windows compatiblity
    # ----
    #if delete_train_dir:
    #    if tf.gfile.Exists(FLAGS.train_dir):
    #        tf.gfile.Remove(FLAGS.train_dir)
    #    tf.gfile.MakeDirs(FLAGS.train_dir)

    if not tf.gfile.Exists(FLAGS.test_dataset) or \
       not tf.gfile.IsDirectory(FLAGS.test_dataset):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.test_dataset,))

    filenames = tf.gfile.ListDirectory(FLAGS.test_dataset)
    filenames = sorted([os.path.join(FLAGS.test_dataset, f) for f in filenames])

    return filenames


def setup_tensorflow():
    # Create session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction)
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options)
    sess = tf.Session(config=config)

    # Initialize rng with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)
        
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    return sess, summary_writer

def _demo():
    # Load checkpoint
    if not tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.checkpoint_dir,))

    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    filenames = prepare_dirs(delete_train_dir=False)

    # Setup async input queues
    features, labels = srez_input.setup_inputs(sess, filenames)

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
            srez_model.create_model(sess, features, labels)

    # Restore variables from checkpoint
    saver = tf.train.Saver()
    filename = 'checkpoint_new.txt'
    filename = os.path.join(FLAGS.checkpoint_dir, filename)
    saver.restore(sess, filename)

    # Execute demo
    srez_demo.demo1(sess)

class TrainData(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

def _train():
    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    all_filenames = prepare_dirs(delete_train_dir=False)

    # Separate training and test sets

    if FLAGS.specific_test:
	train_filenames = all_filenames[:]
        test_filenames = prepare_test_dirs()[:]
    else:
	train_filenames = all_filenames[:-FLAGS.test_vectors]
        test_filenames  = all_filenames[-FLAGS.test_vectors:]


    # Setup async input queues
    train_features, train_labels = srez_input.setup_inputs(sess, train_filenames)
    test_features,  test_labels  = srez_input.setup_inputs(sess, test_filenames)

    # Add some noise during training (think denoising autoencoders)
    noise_level = .03
    noisy_train_features = train_features + \
                           tf.random_normal(train_features.get_shape(), stddev=noise_level)

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
            srez_model.create_model(sess, noisy_train_features, train_labels)

    gene_loss, gene_l1_loss, gene_ce_loss = srez_model.create_generator_loss(disc_fake_output, gene_output, train_features, train_labels)
    disc_real_loss, disc_fake_loss = \
                     srez_model.create_discriminator_loss(disc_real_output, disc_fake_output)
    disc_loss = tf.add(2*FLAGS.disc_real_factor*disc_real_loss, 2*(1-FLAGS.disc_real_factor)*disc_fake_loss, name='disc_loss')
    
    (global_step, learning_rate, gene_minimize, disc_minimize) = \
            srez_model.create_optimizers(gene_loss, gene_var_list,
                                         disc_loss, disc_var_list)
    # Train model
    train_data = TrainData(locals())
    srez_train.train_model(train_data)

def main(argv=None):
    if FLAGS.run == 'demo':
        _demo()
    elif FLAGS.run == 'train':
        _train()

if __name__ == '__main__':
    tf.app.run()
