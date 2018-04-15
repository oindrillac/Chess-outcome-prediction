import pandas
import numpy as np
import matplotlib.pyplot as plt
import chess
from sklearn import svm
from sklearn.model_selection import cross_val_score as cvs
import sys

DEFAULT_FILEPATH = "../../games.csv"
SQUARE_VECTOR = ['P', 'N', 'B', 'R', 'K', 'Q', 'p', 'n', 'b', 'r', 'k', 'q', None]
BOARD_SIZE = 64
DATA_MAX = 200
WHITE_WIN = 0
BLACK_WIN = 1
DRAW_LABEL = 2

def getBoardVector(board):
   """Translate an ASCII board to a long vector.
   Parameters
   ----------
   board : chess.Board
   Returns
   -------
   result : List[int]
      The result list can be divided into sublist of length len(SQUARE_VECTOR).
      Each sublist is the category vector representing a square on the board.
   """
   result = [0] * (BOARD_SIZE * len(SQUARE_VECTOR))
   for i in range(BOARD_SIZE):
      if not board.piece_at(i):
         result[i * len(SQUARE_VECTOR) + SQUARE_VECTOR.index(None)] = 1
      else:
         result[i * len(SQUARE_VECTOR) + SQUARE_VECTOR.index(board.piece_at(i).symbol())] = 1
   return result

def getData(filepath, data_max):
   """Reads the data.
   Parameters
   ----------
   filepath : string
   data_max : int
      The maximum amount of data to analyze.
   Returns
   -------
   boards : List[List[int]]
      Each element is a long list representation of a board.
   labels : List[
   """
   try:
      f = open(filepath, 'r')
      f.close()
   except:
      exit("File " + filepath + " not found.")
   df = pandas.read_csv(filepath)
   movesAll = df['moves'].values.tolist()
   labelsString = df['winner'].values.tolist()
   labelsString = labelsString[:data_max]
   boards = []
   for i in range(len(labelsString)):
      moves = movesAll[i].split()
      board = chess.Board()
      for m in moves:
         board.push_san(m)
      boards.append(getBoardVector(board))
   
   labels = [-1] * len(labelsString)
   for i in range(len(labels)):
      if labelsString[i] == 'white':
         labels[i] = 0
      elif labelsString[i] == 'black':
         labels[i] = 1
      else:
         labels[i] = 2
   return boards, labels

def classification(data, labels):
   clf = svm.SVC(kernel = 'poly', C = 1, degree = 6, probability = True)
   scores = cvs(clf, data, labels, cv = 3)
   print(scores)

if __name__ == '__main__':
   filepath = DEFAULT_FILEPATH
   data_max = DATA_MAX
   if len(sys.argv) > 1:
      filepath = sys.argv[1]
      if len(sys.argv) > 2:
         data_max = int(sys.argv[2])
   boards, labels = getData(filepath, data_max)
   classification(boards, labels)