import os
import scipy
import tensorflow as tf
import srez_main
import srez_input
import srez_model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_samples', 1,
                            "Number of samples to test")

# Load checkpoint folder
if not tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
	raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.checkpoint_dir,))

# Setup global tensorflow state
sess, summary_writer = srez_main.setup_tensorflow()

# Prepare directories
filenames = srez_main.prepare_test_dirs()
assert len(filenames) >= FLAGS.max_samples , "Not enough test images"

# Setup async input queues
features, labels = srez_input.setup_inputs(sess, filenames)

# Create and initialize model
[gene_minput, gene_moutput,
gene_output, gene_var_list,
disc_real_output, disc_fake_output, disc_var_list] = \
    srez_model.create_model(sess, features, labels)

# Restore variables from checkpoint
saver = tf.train.Saver()
ckpt_filename = 'checkpoint_new.txt'
ckpt_filepath = os.path.join(FLAGS.checkpoint_dir, ckpt_filename)
saver.restore(sess, ckpt_filepath)

# Run inference using pretrained model
feature, label = sess.run([features, labels])

feed_dict = {gene_minput: feature}
gene_output = sess.run(gene_moutput, feed_dict=feed_dict)

size = [label.shape[1], label.shape[2]]

nearest = tf.image.resize_nearest_neighbor(feature, size)
nearest = tf.maximum(tf.minimum(nearest, 1.0), 0.0)

bicubic = tf.image.resize_bicubic(feature, size)
bicubic = tf.maximum(tf.minimum(bicubic, 1.0), 0.0)

clipped = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)
resized = tf.image.resize_nearest_neighbor(clipped, size)

image   = tf.concat([nearest, resized, label], 2)

image = image[0:FLAGS.max_samples,:,:,:]
image = tf.concat([image[i,:,:,:] for i in range(FLAGS.max_samples)], 0)
image = sess.run(image)

result_filename = 'test_result.png'
scipy.misc.toimage(image, cmin=0., cmax=1.).save(result_filename)
print("    Saved test result")