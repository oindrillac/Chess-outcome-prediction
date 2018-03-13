import argparse
import h5py
import pandas
import chess

MOVES_KEY = 'moves'
WINNER_KEY = 'winner'
DEFAULT_BOARD_DATASET_NAME = 'boards'
DEFAULT_LABEL_DATASET_NAME = 'labels'
BOARD_SIZE = 64
ONE_HOT_VECTOR_SIZE = 13
MAX_NUM_BOARDS = 10 # TODO increase this.
NUM_LABELS = 3
WHITE_WIN_LABEL = [1, 0, 0]
BLACK_WIN_LABEL = [0, 1, 0]
DRAW_LABEL = [0, 0, 1]

# How many of the first generated boards to skip.
SKIP_FIRST_N_MOVES = 1

# The chess API represents white pieces by uppercase letters and black
# pieces by lowercase letters. This constant translates the symbols
# used to represent the pieces into indices for one-hot vectors.
ONE_HOT_INDICES = ['P', 'N', 'B', 'R', 'K', 'Q', 'p', 'n', 'b', 'r', 'k', 'q', None]


def get_one_hot_vector(chess_piece):
    """Returns a 13-dimensional one-hot vector that represents this
    piece. Input is a chess.Piece, or None if square is empty."""
    vector = [0] * ONE_HOT_VECTOR_SIZE
    if not chess_piece:
        vector[ONE_HOT_INDICES.index(None)] = 1
    else:
        vector[ONE_HOT_INDICES.index(chess_piece.symbol())] = 1
    return vector


def get_vectors(chess_board):
    """Returns a matrix of size 64x13 where each row represents a
    square (in A1, A2, ..., G8, H8 order--that is, row major order
    starting from white's rear rank and ending with black's rear rank).
    The content of each row is a one-hot vector that represents which
    piece is on that square. The input is a chess.Board."""
    return [get_one_hot_vector(chess_board.piece_at(i)) for i in range(BOARD_SIZE)]


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
    dataset_winners = games[WINNER_KEY]
    dataset_moves = games[MOVES_KEY]
    if num_games:
        dataset_winners = dataset_winners[:num_games]
        dataset_moves = dataset_moves[:num_games]
    for i in range(len(dataset_winners)):
        if label_dataset.shape[0] >= MAX_NUM_BOARDS:
            break
        # Get label.
        label = get_label(dataset_winners[i])
        # Get moves.
        game_moves = dataset_moves[i].split(' ')
        # Generate boards.
        board = chess.Board()
        game_boards = [board.copy()]
        for move in game_moves:
            board.push_san(move)
            game_boards.append(board.copy())
        # Keep only boards with white-to-play.
        game_boards = game_boards[::2]
        # Skip the first few boards.
        game_boards = game_boards[SKIP_FIRST_N_MOVES:]
        end = min(MAX_NUM_BOARDS - board_dataset.shape[0], len(game_boards))
        for i in range(end):
            board_vectors = get_vectors(game_boards[i])
            board_dataset.resize((board_dataset.shape[0] + 1, board_dataset.shape[1], board_dataset.shape[2]))
            board_dataset[-1, :, :] = board_vectors 
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
    make_dataset(args.csv_filename, args.output_filename, num_games=args.num_games)

