"""
recurrent.py

Skeleton file for the Recurrent Neural Network for Penn Treebank Language Modeling.

Creates and trains the model, then evaluates performance on the test set.

Run via: `python recurrent.py`
"""
import numpy as np
import pickle
import tensorflow as tf 

# Fetch Datasets from Pickled Files
with open("data/processed_data/train.pik", 'r') as f:
    trainX, trainY, vocab = pickle.load(f)
with open("data/processed_data/test.pik", 'r') as g:
    testX, testY, _ = pickle.load(g)

# Setup the Model Parameters (Layer sizes, Batch Size, etc.)
EMBEDDING_SIZE, LSTM_SIZE, VOCAB_SIZE = 30, 256, len(vocab)
NUM_EPOCHS, BATCH_SIZE, WINDOW_SIZE = 1, 50, 20

### Start Building the Computation Graph ###

# Initializer - initialize our variables from standard normal with stddev 0.1
initializer = tf.random_normal_initializer(stddev=0.1)

# Setup Placeholders => None argument in shape lets us pass in arbitrary sized batches
X = tf.placeholder(tf.int32, shape=[None, WINDOW_SIZE])  
Y = tf.placeholder(tf.int32, shape=[None, WINDOW_SIZE])
keep_prob = tf.placeholder(tf.float32)

######################## FILL ME IN ###########################
# Embedding Matrix
E = tf.get_variable("Embedding", shape=[VOCAB_SIZE, EMBEDDING_SIZE], initializer=initializer)

# Embedding Lookup + Dropout
embeddings = tf.nn.embedding_lookup(E, X)               # Shape: [None, WINDOW_SZ, EMBED_SZ]
drop_embeddings = tf.nn.dropout(embeddings, keep_prob)

# Basic LSTM Cell, Initial State 
lstm = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE)
initial_state = lstm.zero_state(BATCH_SIZE, tf.float32)

# Run the LSTM over Inputs to get Outputs, State 
outputs, final_state = tf.nn.dynamic_rnn(lstm, drop_embeddings, initial_state=initial_state)

# Output Layer Variables
W_1 = tf.get_variable("Output_W", shape=[LSTM_SIZE, VOCAB_SIZE], initializer=initializer)
b_1 = tf.get_variable("Output_b", shape=[VOCAB_SIZE], initializer=initializer)

# Output Layer Transformation => Outputs is 3D Tensor, so use tensordot
output = tf.tensordot(outputs, W_1, axes=[[2],[0]]) + b_1
###############################################################

# Compute Loss
loss = tf.contrib.seq2seq.sequence_loss(output, Y, tf.ones([BATCH_SIZE, WINDOW_SIZE]))

# Setup Optimizer
train_op = tf.train.AdamOptimizer().minimize(loss)

### Launch the Session, to Communicate with Computation Graph ###
with tf.Session() as sess:
    # Initialize all variables in the graph
    sess.run(tf.global_variables_initializer())

    # Training Loop
    for i in range(NUM_EPOCHS):
        chunk, state, counter = BATCH_SIZE * WINDOW_SIZE, sess.run(initial_state), 0
        for (start, end) in zip(range(0, len(trainX), chunk), range(chunk, len(trainX), chunk)):
            batch_x = trainX[start:end].reshape([BATCH_SIZE, WINDOW_SIZE])
            batch_y = trainY[start:end].reshape([BATCH_SIZE, WINDOW_SIZE])
            curr_cost, state, _ = sess.run([loss, final_state, train_op], 
                                           feed_dict={X: batch_x, Y: batch_y, initial_state: state, keep_prob: 0.5})
            if counter % 100 == 0:
                print "Step %d Loss: %.3f, Perplexity: %.3f" % (counter, curr_cost, np.exp(curr_cost))
            counter +=1
    
    # Evaluate on Test Data
    state, counter, sum_cost = sess.run(initial_state), 0, 0.0
    for (start, end) in zip(range(0, len(testX), chunk), range(chunk, len(testX), chunk)):
        batch_x = testX[start:end].reshape([BATCH_SIZE, WINDOW_SIZE])
        batch_y = testY[start:end].reshape([BATCH_SIZE, WINDOW_SIZE])
        cost, state = sess.run([loss, final_state], feed_dict={X: batch_x, Y: batch_y, 
                                                               initial_state: state, keep_prob: 1.0})
        sum_cost, counter = sum_cost + cost, counter + 1
    print "Test Perplexity: %.3f" % np.exp(sum_cost / counter)
