import argparse
import h5py
import pandas

MOVES_KEY = 'moves'
WINNER_KEY = 'winner'
DEFAULT_BOARD_DATASET_NAME = 'boards'
DEFAULT_LABEL_DATASET_NAME = 'label'
BOARD_SIZE = 64
ONE_HOT_VECTOR_SIZE = 13
MAX_NUM_BOARDS = 10 # TODO increase this.
NUM_LABELS = 3
WHITE_WIN_LABEL = [1, 0, 0]
BLACK_WIN_LABEL = [0, 1, 0]
DRAW_LABEL = [0, 0, 1]


def get_label(winner):
    """Returns the appropriate label vector based on the winner string."""
    if winner == 'white':
        return WHITE_WIN_LABEL
    elif winner == 'black':
        return BLACK_WIN_LABEL
    elif winner == 'draw':
        return DRAW_LABEL
    else:
        raise ValueError('Unrecognized winner string: {0}'.format(winner))


def make_dataset(csv_filename, output_filename, num_games=None):
    """Builds the compiled dataset and saves it to output_filename.
    The output file will be an h5 file that contains the one-hot
    vectors to be used in training. If num_games is None, then this
    function will process all games in the dataset."""
    outfile = h5py.File(output_filename, 'w')
    board_dataset = outfile.create_dataset(DEFAULT_BOARD_DATASET_NAME, (0, BOARD_SIZE, ONE_HOT_VECTOR_SIZE), maxshape=(MAX_NUM_BOARDS, BOARD_SIZE, ONE_HOT_VECTOR_SIZE), chunks=True, dtype='i1')
    label_dataset = outfile.create_dataset(DEFAULT_LABEL_DATASET_NAME, (0, NUM_LABELS), maxshape=(MAX_NUM_BOARDS, NUM_LABELS), chunks=True, dtype='i1')
    games = pandas.read_csv(csv_filename)        
    if num_games:
        games = games[:num_games]
    for game in games:
        # Get moves.
        moves = game[MOVES_KEY]
        # Generate boards.
        # TODO generate boards. board is a matrix of size 64x13 where each row is a one-hot vector that tells which piece is on that board.
        boards = []
        for board in boards:
            board_dataset.resize((board_dataset.shape[0] + 1, board_dataset.shape[1], board_dataset.shape[2]))
            board_dataset[-1, :, :] = board 
        # Get label.
        winner = game[WINNER_KEY]
        label = get_label(winner)
        label_dataset.resize((label_dataset.shape[0] + 1, label_dataset.shape[1]))
        label_dataset[-1] = label
    outfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates the compiled one-hot vector dataset from the input CSV games file found on Kaggle.')
    parser.add_argument('csv_filename', metavar='f', type=str,
        help='The name of the CSV games file to use (if using the Kaggle dataset, this should be the path to games.csv).')
    parser.add_argument('output_filename', metavar='o', type=str,
        help='The name of the h5 file at which to save the compiled dataset.')
    parser.add_argument('-n', '--num_games', metavar='n', type=int, default=None,
        help='The maximum number of games to process. By default, processes all functions in the file.')
    args = parser.parse_args()

