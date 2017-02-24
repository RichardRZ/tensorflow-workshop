"""
convolution.py

Skeleton file for the 3-Layer Convolutional Neural Network for the MNIST Task.

Creates and trains the model, then evaluates performance on the test set.

Run via: `python convolution.py`
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf 

# Fetch MNIST Dataset using the supplied Tensorflow Utility Function
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Setup the Model Parameters (Layer sizes, Batch Size, etc.)
INPUT_SIZE, OUTPUT_SIZE = 784, 10  
FILTER_SIZE, FILTER_ONE_DEPTH, FILTER_TWO_DEPTH = 5, 32, 64
FLAT_SIZE, HIDDEN_SIZE = 7 * 7 * 64, 1024
BATCH_SIZE, NUM_TRAINING_STEPS = 100, 1000

# Create Convolution/Pooling Helper Functions 
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

### Start Building the Computation Graph ###

# Initializer - initialize our variables from standard normal with stddev 0.1
initializer = tf.random_normal_initializer(stddev=0.1)

# Setup Placeholders => None argument in shape lets us pass in arbitrary sized batches
X = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])  
Y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])
keep_prob = tf.placeholder(tf.float32) 

######################## FILL ME IN ###########################
# Reshape input so it resembles an image (height x width x depth)
X_image = tf.reshape(X, [-1, 28, 28, 1])

# Conv Filter 1 Variables
Wconv_1 = tf.get_variable("WConv_1", shape=[FILTER_SIZE, FILTER_SIZE, 1, FILTER_ONE_DEPTH], initializer=initializer)
bconv_1 = tf.get_variable("bConv_1", shape=[FILTER_ONE_DEPTH], initializer=initializer)

# First Convolutional + Pooling Transformation
h_conv1 = tf.nn.relu(conv2d(X_image, Wconv_1) + bconv_1)
h_pool1 = max_pool_2x2(h_conv1)

# Conv Filter 2 Variables
Wconv_2 = tf.get_variable("WConv_2", shape=[FILTER_SIZE, FILTER_SIZE, FILTER_ONE_DEPTH, FILTER_TWO_DEPTH], 
                          initializer=initializer)
bconv_2 = tf.get_variable("bConv_2", shape=[FILTER_TWO_DEPTH], initializer=initializer)

# Second Convolutional + Pooling Transformation
h_conv2 = tf.nn.relu(conv2d(h_pool1, Wconv_2) + bconv_2)
h_pool2 = max_pool_2x2(h_conv2)

# Flatten Convolved Image, into vector for remaining feed-forward transformations
h_pool2_flat = tf.reshape(h_pool2, [-1, FLAT_SIZE])

# Hidden Layer Variables
W_1 = tf.get_variable("W_1", shape=[FLAT_SIZE, HIDDEN_SIZE], initializer=initializer)
b_1 = tf.get_variable("b_1", shape=[HIDDEN_SIZE], initializer=initializer)

# Hidden Layer Transformation
hidden = tf.nn.relu(tf.matmul(h_pool2_flat, W_1) + b_1)

# DROPOUT - For regularization
hidden_drop = tf.nn.dropout(hidden, keep_prob)

# Output Layer Variables
W_2 = tf.get_variable("W_2", shape=[HIDDEN_SIZE, OUTPUT_SIZE], initializer=initializer)
b_2 = tf.get_variable("b_2", shape=[OUTPUT_SIZE], initializer=initializer)

# Output Layer Transformation
output = tf.matmul(hidden_drop, W_2) + b_2
###############################################################

# Compute Loss
loss = tf.losses.softmax_cross_entropy(Y, output)

# Compute Accuracy
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(output, 1))
accuracy = 100 * tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Setup Optimizer
train_op = tf.train.AdamOptimizer().minimize(loss)

### Launch the Session, to Communicate with Computation Graph ###
with tf.Session() as sess:
    # Initialize all variables in the graph
    sess.run(tf.global_variables_initializer())

    # Training Loop
    for i in range(NUM_TRAINING_STEPS):
        batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
        curr_acc, _ = sess.run([accuracy, train_op], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
        if i % 100 == 0:
            print 'Step %d Current Training Accuracy: %.3f' % (i, curr_acc)
    
    # Evaluate on Test Data
    print 'Test Accuracy: %.3f' % sess.run(accuracy, feed_dict={X: mnist.test.images, 
                                                                Y: mnist.test.labels,
                                                                keep_prob: 1.0})
