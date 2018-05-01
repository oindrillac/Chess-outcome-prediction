"""This model is based on the TensorFlow MNIST example. The underlying
assumption of this model is that we can treat the input boards like
images."""
from __future__ import print_function
import argparse
import h5py
import numpy as np
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt
import sys

sys.path.extend(['.', '..'])

import build_dataset
import train_mnist

tf.logging.set_verbosity(tf.logging.INFO)


def main(argv):
    """Tests the model using the given argv."""
    if len(argv) != 3:
        raise ValueError('Expected 3 arguments; got {0}.'.format(len(argv)))
    dataset_filename = argv[1]
    model_output_dir = argv[2]
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
            if train_mnist.MODIFY_ONE_HOT_INDICES:
                flat_value = train_mnist.convert_index_value(flat_value)
            boards_flat[example_number][board_position] = flat_value
    labels_flat = np.zeros((labels.shape[0]), dtype=np.int32)
    for example_number in range(len(labels)):
        flat_value = np.where(labels[example_number] == 1)[0][0]
        labels_flat[example_number] = flat_value
    if train_mnist.NUM_CLASSES == 2:
        boards_binary = []
        labels_binary = []
        for i in range(len(labels_flat)):
            if labels_flat[i] == 0 or labels_flat[i] == 1:
                boards_binary.append(boards_flat[i])
                labels_binary.append(labels_flat[i])
        boards_flat = np.asarray(boards_binary)
        labels_flat = np.asarray(labels_binary)
    split_index = int(len(boards_flat) * train_mnist.TRAINING_SPLIT)
    boards_train = boards_flat[:split_index]
    labels_train = labels_flat[:split_index]
    boards_eval = boards_flat[split_index:]
    labels_eval = labels_flat[split_index:]
    if train_mnist.EVAL_ON_SAME_MATERIAL_BOARDS:
        print('Evaluating on boards with the same material value.')
        piece_values = [0, -8, -5, -3, -3, -1, 0, 1, 3, 3, 5, 8, 0]
        new_boards_eval = []
        new_labels_eval = []
        for i in range(len(boards_eval)):
            board_values = []
            for j in range(len(boards_eval[i])):
                # Map black to negative values.
                # Map white to positive values.
                index_value = int((boards_eval[i][j] + 1) / train_mnist.PIXEL_VALUE_SCALING_FACTOR)
                board_values.append(piece_values[index_value])
            if sum(board_values) == 0:
                new_boards_eval.append(boards_eval[i])
                new_labels_eval.append(labels_eval[i])
        print('Number of eval boards before: {0}'.format(len(boards_eval)))
        boards_eval = np.array(new_boards_eval)
        labels_eval = np.array(new_labels_eval)
        print('Number of eval boards after: {0}'.format(len(new_boards_eval)))
    # Create the Estimator.
    mnist_classifier = tf.estimator.Estimator(model_fn=train_mnist.cnn_model_fn, model_dir=model_output_dir)
    # Set up logging to STDERR.
    tensors_to_log = {}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=train_mnist.LOG_EVERY_N_ITER)
    # Evaluate the model and print results.
    eval_train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': boards_train},
        y=labels_train,
        num_epochs=1,
        shuffle=False)
    eval_train_results = mnist_classifier.evaluate(input_fn=eval_train_input_fn, name='Training Performance')
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
    probabilities = [d['probabilities'] for d in predict_test_results]
    for i in range(len(predicted_classes) - train_mnist.NUM_BOARDS_TO_ANALYZE, len(predicted_classes)):
        print('Board {0}: class = {1}, probability = {2}'.format(i, predicted_classes[i], probabilities[i]))
        train_mnist.print_board(boards_eval[i])
    if len(labels_eval) != len(predicted_classes):
        raise ValueError('Predictions ({0}) and labels ({1}) are of different length.'.format(len(predicted_classes), len(labels_eval)))
    for c in range(train_mnist.NUM_CLASSES):
        print('Number of class {0} labels in the training dataset: {1}'.format(c, sum([1 if x == c else 0 for x in labels_train])))
        print('Number of class {0} labels in the testing dataset: {1}'.format(c, sum([1 if x == c else 0 for x in labels_eval])))
    if train_mnist.NUM_CLASSES == 2:
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
        plt.savefig('{0}roc.png'.format(model_output_dir), format='png')
        print('ROC Curve saved to {0}roc.png'.format(model_output_dir))
        
        precision, recall, thresholds = metrics.precision_recall_curve(labels_eval, scores_eval)
        plt.figure()
        plt.plot(recall, precision, color='darkorange', label='PR Curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve (0 = White Wins, 1 = Black Wins)')
        plt.legend(loc="lower left")
        plt.savefig('{0}pr.png'.format(model_output_dir), format='png')
        print('PR Curve saved to {0}pr.png'.format(model_output_dir))
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


def test_model(dataset_filename, model_output_dir):
    """Tests a model using the given dataset."""
    tf.app.run(main=main, argv=[sys.argv[0]] + [dataset_filename, model_output_dir])


if __name__ == '__main__':
    print('Test this model by running the model.py script with the appropriate arguments.')

