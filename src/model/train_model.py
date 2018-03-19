import argparse

MNIST_MODEL_NUM = 0


def train_model(model_num, dataset_filename, output_filename):
    """Trains a model using the given dataset and saves it to the
    output file. Model numbers are defined above as constants."""
    if model_num == MNIST_MODEL_NUM:
        # TODO call MNIST model.
        pass
    else:
        raise ValueError('Unrecognized model number: {0}'.format(model_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs the model.')
    parser.add_argument('model_num', metavar='m', type=int,
        help='Defines the type of model to train. Each type solves the problem using its own approach.')
    parser.add_argument('dataset_filename', metavar='d', type=str,
        help='The name of the h5 file from which to load the dataset.')
    parser.add_argument('output_filename', metavar='o', type=str,
        help='The name of the h5 file at which to save the trained model.')
    args = parser.parse_args()
    train_model(args.model_num, args.dataset_filename, args.output_filename)

