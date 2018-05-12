"""This model is based on the TensorFlow MNIST example. The underlying
assumption of this model is that we can treat the input boards like
images."""
import argparse
import h5py
import numpy as np
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt
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
DROPOUT_RATE = 0.4
# If True, this will change the one-hot index values to be more analogous to a pixel value.
MODIFY_ONE_HOT_INDICES = True
NUM_CLASSES = 2
PIXEL_VALUE_SCALING_FACTOR = 255 / 12

tf.logging.set_verbosity(tf.logging.INFO)


def convert_index_value(i):
    """Returns a new one-hot index that more closely matches a pixel
    representation of the label. Empty space is mapped to the middle
    value, and on either sides piece indices are reflected with the
    highest values furthest from center."""
    piece = build_dataset.ONE_HOT_INDICES[i]
    result_indices = ['K', 'Q', 'R', 'B', 'N', 'P', None, 'p', 'n', 'b', 'r', 'q', 'k']
    return result_indices.index(piece) * PIXEL_VALUE_SCALING_FACTOR


def mlp_model_fn(features, labels, mode):
    """Model function for CNN. Adapted from TF MNIST tutorial."""
    # Convert 13 dimensional one-hot vector into a flat value (just use the index as "pixel" value). Then we can go directly off the example.
    features_flat = features['x']
    # Input Layer.
    # Input is batch_size x 8 x 8 x 1, where each element is in [0, 12].
    input_layer = tf.reshape(features_flat, [-1, build_dataset.BOARD_ROWS, build_dataset.BOARD_COLS, 1])
    tf.summary.image('First_Boards_In_Batch', input_layer, 10)

    # Hidden fully connected Layer #1.
    # Output is batch_size x 8 x 8 x 32.
    hid1 = tf.layers.dense(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)
    tf.summary.histogram('Hidden1', hid1)


    # Dense Layer.
    # Output is batch_size x 256.
    dense = tf.layers.dense(inputs= hid1, units=256, activation=tf.nn.relu)
    tf.summary.histogram('Dense', dense)
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=DROPOUT_RATE,
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
    if NUM_CLASSES == 2:
        boards_binary = []
        labels_binary = []
        for i in range(len(labels_flat)):
            if labels_flat[i] == 0 or labels_flat[i] == 1:
                boards_binary.append(boards_flat[i])
                labels_binary.append(labels_flat[i])
        boards_flat = np.asarray(boards_binary)
        labels_flat = np.asarray(labels_binary)
    split_index = int(len(boards_flat) * TRAINING_SPLIT)
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
    mnist_classifier.train(input_fn=train_input_fn, hooks=None if not tensors_to_log else [logging_hook])
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
    # Run prediction on the test set.
    predict_test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': boards_eval},
        num_epochs=1,
        shuffle=False)
    predict_test_results = list(mnist_classifier.predict(predict_test_input_fn))
    predicted_classes = [d['classes'] for d in predict_test_results]
    if len(labels_eval) != len(predicted_classes):
        raise ValueError('Predictions ({0}) and labels ({1}) are of different length.'.format(len(predicted_classes), len(labels_eval)))
    for c in range(NUM_CLASSES):
        print('Number of class {0} labels in the training dataset: {1}'.format(c, sum([1 if x == c else 0 for x in labels_train])))
        print('Number of class {0} labels in the testing dataset: {1}'.format(c, sum([1 if x == c else 0 for x in labels_eval])))
    if NUM_CLASSES == 2:
        precision = metrics.precision_score(labels_eval, predicted_classes)
        recall = metrics.recall_score(labels_eval, predicted_classes)
        f1_score = metrics.f1_score(labels_eval, predicted_classes)
        roc_auc = metrics.roc_auc_score(labels_eval, predicted_classes)
        confusion = metrics.confusion_matrix(labels_eval, predicted_classes)
        scores_eval = [d['probabilities'][1] for d in predict_test_results]
        fpr, tpr, thresholds = metrics.roc_curve(labels_eval, scores_eval)
        print('Precision: {0}'.format(precision))
        print('Recall: {0}'.format(recall))
        print('F1 Score: {0}'.format(f1_score))
        print('ROC AUC: {0}'.format(roc_auc))
        print('Confusion matrix:\n{0}'.format(confusion))

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label='ROC Curve')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Baseline')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (0 = White Wins, 1 = Black Wins)')
        plt.legend(loc="lower right")
        plt.savefig('../../roc.png', format='png')
        print('ROC Curve saved to ../../roc.png')
        
        precision, recall, thresholds = metrics.precision_recall_curve(labels_eval, scores_eval)
        plt.figure()
        plt.plot(recall, precision, color='darkorange', label='PR Curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve (0 = White Wins, 1 = Black Wins)')
        plt.legend(loc="lower left")
        plt.savefig('../../pr.png', format='png')
        print('PR Curve saved to ../../pr.png')
    else:
        precision = metrics.precision_score(labels_eval, predicted_classes, average='weighted')
        recall = metrics.recall_score(labels_eval, predicted_classes, average='weighted')
        f1_score = metrics.f1_score(labels_eval, predicted_classes, average='weighted')
        confusion = metrics.confusion_matrix(labels_eval, predicted_classes)
        print('Precision: {0}'.format(precision))
        print('Recall: {0}'.format(recall))
        print('F1 Score: {0}'.format(f1_score))
        print('Confusion matrix:\n{0}'.format(confusion))

    infile.close()


def train_model(dataset_filename, output_filename, batch_size='', num_epochs=''):
    """Trains a model using the given dataset and saves it to the
    output file."""
    tf.app.run(main=main, argv=[sys.argv[0]] + [dataset_filename, output_filename, batch_size, num_epochs])


if __name__ == '__main__':
    print 'Train this model by running the train_model.py script with the appropriate arguments.'
