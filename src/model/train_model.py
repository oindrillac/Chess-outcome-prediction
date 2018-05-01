import argparse
import train_mnist

MNIST_MODEL_NUM = 0


def train_model(model_num, dataset_filename, output_filename, model_directory, batch_size, num_epochs):
    """Trains a model using the given dataset and saves it to the
    output file. Model numbers are defined above as constants."""
    if model_num == MNIST_MODEL_NUM:
        train_mnist.train_model(dataset_filename, output_filename, model_directory, batch_size=batch_size, num_epochs=num_epochs)
    else:
        raise ValueError('Unrecognized model number: {0}'.format(model_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains the model.')
    parser.add_argument('model_num', metavar='m', type=int,
        help='Defines the type of model to train. Each type solves the problem using its own approach.')
    parser.add_argument('dataset_filename', metavar='d', type=str,
        help='The name of the h5 file from which to load the dataset.')
    parser.add_argument('output_filename', metavar='o', type=str,
        help='The name of the h5 file at which to save the trained model.')
    parser.add_argument('model_directory', metavar='m', type=str,
        help='The name of the directory to which to save logs.')
    parser.add_argument('-b', '--batch_size', metavar='b', type=str, default='',
        help='The batch size to use in the model (if the model uses batching). By default, uses model-specific default value.')
    parser.add_argument('-e', '--num_epochs', metavar='e', type=str, default='',
        help='The number of epochs to train the model (if the model uses epochs). By default, uses model-specific default value.')
    args = parser.parse_args()
    train_model(args.model_num, args.dataset_filename, args.output_filename, args.model_directory, args.batch_size, args.num_epochs)

