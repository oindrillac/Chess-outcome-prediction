"""This model is based on the TensorFlow MNIST example. The underlying
assumption of this model is that we can treat the input boards like
images."""
import argparse
import h5py
import numpy as np
import tensorflow as tf
import sys

sys.path.extend(['.', '..'])

import build_dataset

DEFAULT_MODEL_DIR = '/tmp/mnist_convnet_model'
# For logging and summaries.
LOG_EVERY_N_ITER = 50
DEFAULT_BATCH_SIZE = 100
DEFAULT_NUM_EPOCHS = 5
TRAINING_SPLIT = 0.8
LEARNING_RATE = 0.001
# If True, this will change the one-hot index values to be more analogous to a pixel value.
MODIFY_ONE_HOT_INDICES = True

tf.logging.set_verbosity(tf.logging.INFO)


def convert_index_value(i):
    """Returns a new one-hot index that more closely matches a pixel
    representation of the label. Empty space is mapped to the middle
    value, and on either sides piece indices are reflected with the
    highest values furthest from center."""
    piece = build_dataset.ONE_HOT_INDICES[i]
    result_indices = ['K', 'Q', 'R', 'B', 'N', 'P', None, 'p', 'n', 'b', 'r', 'q', 'k']
    return result_indices.index(piece)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN. Adapted from TF MNIST tutorial."""
    # Convert 13 dimensional one-hot vector into a flat value (just use the index as "pixel" value). Then we can go directly off the example.
    features_flat = features['x']
    # Input Layer.
    # Input is batch_size x 8 x 8 x 1, where each element is in [0, 12].
    input_layer = tf.reshape(features_flat, [-1, build_dataset.BOARD_ROWS, build_dataset.BOARD_COLS, 1])
    tf.summary.image('First_Boards_In_Batch', input_layer, 10)

    # Convolutional Layer #1.
    # Output is batch_size x 8 x 8 x 32.
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)
    tf.summary.histogram('Conv1', conv1)

    # Pooling Layer #1.
    # Output is batch_size x 4 x 4 x 32.
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    tf.summary.histogram('Pool1', pool1)

    # Convolutional Layer #2.
    # Output is batch_size x 4 x 4 x 64.
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)
    tf.summary.histogram('Conv2', conv2)

    # Pooling Layer #2.
    # Output is batch_size x 2 x 2 x 64.
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    tf.summary.histogram('Pool2', pool2)

    # Dense Layer.
    # Output is batch_size x 256.
    pool2_flat = tf.reshape(pool2, [-1, 2 * 2 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu)
    tf.summary.histogram('Dense', dense)
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN)
    tf.summary.histogram('Dropout', dropout)

    # Logits Layer.
    # Output is batch_size x 3.
    logits = tf.layers.dense(inputs=dropout, units=3)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode).
        'classes': tf.argmax(input=logits, axis=1),
        # Add softmax_tensor to the graph. It is used for PREDICT and by the logging_hook.
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss and accuracy (for both TRAIN and EVAL modes).
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode).
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        tf.summary.scalar('Training_Loss', loss)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode).
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
    confusion = tf.confusion_matrix(labels, predictions['classes'])
    eval_metric_ops = {'accuracy': accuracy}
    tf.summary.scalar('Testing_Loss', loss)
    tf.summary.scalar('Testing_Accuracy', accuracy)
    tf.summary.tensor_summary('Confusion_Matrix', confusion)
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(argv):
    """Trains the model using the given argv."""
    if len(argv) != 5:
        raise ValueError('Expected 5 arguments; got {0}.'.format(len(argv)))
    dataset_filename = argv[1]
    output_filename = argv[2]
    batch_size = DEFAULT_BATCH_SIZE if not argv[3] else int(argv[3])
    num_epochs = DEFAULT_NUM_EPOCHS if not argv[4] else int(argv[4])
    # Load training and eval data.
    infile = h5py.File(dataset_filename, 'r')
    boards = np.asarray(infile[build_dataset.DEFAULT_BOARD_DATASET_NAME])
    labels = np.asarray(infile[build_dataset.DEFAULT_LABEL_DATASET_NAME])
    # Change boards and labels from one hot encoding to flat encoding.
    boards_flat = np.zeros((boards.shape[0], boards.shape[1]), dtype=np.float32)
    for example_number in range(len(boards)):
        for board_position in range(len(boards[example_number])):
            flat_value = np.where(boards[example_number][board_position] == 1)[0][0]
            # Get one-hot indices more analogous to an image. 
            if MODIFY_ONE_HOT_INDICES:
                flat_value = convert_index_value(flat_value)
            boards_flat[example_number][board_position] = flat_value
    labels_flat = np.zeros((labels.shape[0]), dtype=np.int32)
    for example_number in range(len(labels)):
        flat_value = np.where(labels[example_number] == 1)[0][0]
        labels_flat[example_number] = flat_value
    split_index = int(len(boards) * TRAINING_SPLIT)
    boards_train = boards_flat[:split_index]
    labels_train = labels_flat[:split_index]
    boards_eval = boards_flat[split_index:]
    labels_eval = labels_flat[split_index:]
    # Create the Estimator.
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=DEFAULT_MODEL_DIR)
    # Set up logging to STDERR.
    tensors_to_log = {}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=LOG_EVERY_N_ITER)
    # Train the model.
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': boards_train},
        y=labels_train,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        hooks=None if not tensors_to_log else [logging_hook])
    # Evaluate the model and print results.
    eval_train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': boards_train},
        y=labels_train,
        num_epochs=1,
        shuffle=False)
    eval_train_results = mnist_classifier.evaluate(input_fn=eval_train_input_fn, name='Training Performance')
    print('Training results: {0}'.format(eval_train_results))
    eval_test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': boards_eval},
        y=labels_eval,
        num_epochs=1,
        shuffle=False)
    eval_test_results = mnist_classifier.evaluate(input_fn=eval_test_input_fn, name='Testing Performance')
    print('Eval results: {0}'.format(eval_test_results))
    infile.close()


def train_model(dataset_filename, output_filename, batch_size='', num_epochs=''):
    """Trains a model using the given dataset and saves it to the
    output file."""
    tf.app.run(main=main, argv=[sys.argv[0]] + [dataset_filename, output_filename, batch_size, num_epochs])


if __name__ == '__main__':
    print 'Train this model by running the train_model.py script with the appropriate arguments.'

