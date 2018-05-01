import argparse
import model_mnist

MNIST_MODEL_NUM = 0


def test_model(model_num, dataset_filename, model_directory):
    """Tests a model using the given dataset. Model numbers are defined
    above as constants."""
    if model_num == MNIST_MODEL_NUM:
        model_mnist.test_model(dataset_filename, model_directory)
    else:
        raise ValueError('Unrecognized model number: {0}'.format(model_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests the model.')
    parser.add_argument('model_num', metavar='m', type=int,
        help='Defines the type of model to train. Each type solves the problem using its own approach.')
    parser.add_argument('dataset_filename', metavar='d', type=str,
        help='The name of the h5 file from which to load the dataset.')
    parser.add_argument('model_directory', metavar='m', type=str,
        help='The name of the directory to which to save logs.')
    args = parser.parse_args()
    test_model(args.model_num, args.dataset_filename, args.model_directory)

