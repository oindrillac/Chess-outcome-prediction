import pandas
import numpy as np
import matplotlib.pyplot as plt
import chess
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score as cvs
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import sys
from random import shuffle
import h5py
from tempfile import TemporaryFile

BOARDSMATRIX = "boardsMatrix.npy"
LABELSMATRIX = "labelsMatrix.npy"
H5_DATA_LIST = "h5DataList.txt"
H5_LABELS_LIST = "h5LabelsList.txt"
DEFAULT_FILEPATH = "../../games.csv"
DEFAULT_H5_FILEPATH = "../../boards_601253in.hdf5"
SQUARE_VECTOR = ['P', 'N', 'B', 'R', 'K', 'Q', 'p', 'n', 'b', 'r', 'k', 'q', None]
BOARD_SIZE = 64
DATA_MAX = 2000
WHITE_WIN = 0
BLACK_WIN = 1
DRAW_LABEL = 2
MODES = [i for i in range(1, 11)]
OPTION = None


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
   

def getData(filepath, dataMax):
   """Reads the file and returns the desired data and labels.
   Parameters
   ----------
   filepath : string
   dataMax : int
      The maximum amount of data to analyze.
   Returns
   -------
   boards : List[List[int]]
      Each element is a long list representation of a board.
   labels : List[int]
   """
   try:
      f = open(filepath, 'r')
      f.close()
   except:
      exit("File " + filepath + " not found.")
   df = pandas.read_csv(filepath)
   dataMax = min(dataMax, df.shape[0])
   

   # remove some rows
   if dataMax < df.shape[0]:
      toDrop = [i for i in range(df.shape[0])]
      shuffle(toDrop)
      toDrop = toDrop[dataMax :]
      df.drop(df.index[toDrop])
   
   # read the openings
   openingsAll = df['opening_name'].values.tolist()
   openingMap = {}
   counter = 0
   for opening in openingsAll:
      if not opening in openingMap:
         openingMap[opening] = counter
         counter += 1
   openingAmount = len(openingMap)
   openings = []
   for i in range(dataMax):
      openingVector = [0] * openingAmount
      openingVector[openingMap[openingsAll[i]]] = 1
      openings.append(openingVector)
   
   # read the moves and record the final boards
   movesAll = df['moves'].values.tolist()
   boards = []
   for i in range(dataMax):
      moves = movesAll[i].split()
      board = chess.Board()
      for m in moves:
         board.push_san(m)
      boards.append(getBoardVector(board))
      
   # combine the data
   data = []
   for i in range(dataMax):
      data.append(openings[i] + boards[i])
   
   # read the winners and store as labels
   labelsString = df['winner'].values.tolist()
   labelsString = labelsString[:dataMax]
   labels = [-1] * len(labelsString)
   for i in range(len(labels)):
      if labelsString[i] == 'white':
         labels[i] = 0
      elif labelsString[i] == 'black':
         labels[i] = 1
      else:
         labels[i] = 2
   
   
   return data, labels

def getH5Data(filepath, dataMax = None, option = None):
   """Reads the h5 files and returns the desired data and labels.
   Parameters
   ----------
   filepath : string
   dataMax : int
      The maximum amount of data to analyze.
   option : string
      Whether to use saved data.
   Returns
   -------
   boards : List[List[int]]
      Each element is a long list representation of a board.
   labels : List[int]
   """
   data = []
   labels = []
   boardsMatrix = None
   labelsMatrix = None
   indices = None
   if option == "matrix":
      print("Reading from matrix files.")
      boardsMatrix = np.load(BOARDSMATRIX)
      labelsMatrix = np.load(LABELSMATRIX)
      print("Matrix data loaded, shape = " + str(boardsMatrix.shape))
   elif option is None:
      print("Reading from original h5 files.")
      try:
         f = h5py.File(filepath, 'r')
      except:
         print("File " + filepath + " not found.")
         exit()
      f = h5py.File(filepath, 'r')
      boardsMatrix = np.asarray(f["boards"])
      labelsMatrix = np.asarray(f["labels"])
      # print("Saving numpy matrices with shape = " + str(boardsMatrix.shape) + ".")
      np.save(BOARDSMATRIX, boardsMatrix)
      np.save(LABELSMATRIX, labelsMatrix)
   indices = [i for i in range(boardsMatrix.shape[0])]
   if not dataMax is None:
      print("Randomly selecting " + str(dataMax) + " data.")
      shuffle(indices)
      indices = indices[:dataMax]
   
   for i in indices:
      board = boardsMatrix[i, :, :]
      temp = [0] * (board.shape[0] * board.shape[1])
      
      for j in range(board.shape[0]):
         for k in range(board.shape[1]):
            temp[j * board.shape[1] + k] = board[j, k]
      data.append(temp)
      labels.append(list(labelsMatrix[i, :]).index(1))
   return data, labels
   
def getPreprocessedData(data, basisAmount):
   """Preprocesses the data.
   Parameters
   ----------
   data : List[List[int]]
   """
   dataMatrix = np.matrix(data)
   S, U, V = np.linalg.svd(dataMatrix)
   result = np.dot(dataMatrix, V[:, : basisAmount])
   return result

def getTrainingAccuracy(classifier, data, labels):
   """Performs classifications and obtain training accuracy.
   Parameters
   ----------
   classifier : sklearn classifier
   data : List[List[float]]
   labels : List[int]
   trials : int
      The number of trials of cross-validation.
   Returns
   -------
   trainingAccuracy : float
   """
   classifier.fit(data, labels)
   predicted = classifier.predict(data)
   trainingAccuracy = 0
   for i in range(len(labels)):
      if predicted[i] == labels[i]:
         trainingAccuracy += 1
   trainingAccuracy /= len(labels)
   return trainingAccuracy

def getROCAUCScore(classifier, data, labels, trials = 3, classes = 3):
   """Performs classifications and obtain ROC AUC.
   Parameters
   ----------
   classifier : sklearn classifier
   data : List[List[float]]
   labels : List[int]
   trials : int
      The number of trials of to perform.
   Returns
   -------
   None
   """
   area = 0
   labels = label_binarize(labels, classes = [i for i in range(classes)])
   for trial in range(trials):
      trainingData, testingData, trainingLabels, testingLabels = train_test_split(data, labels, test_size = 0.3)
      labelScores = classifier.fit(trainingData, trainingLabels).decision_function(testingData)
      fprMicro, tprMicro, _ = roc_curve(testingLabels.ravel(), labelScores.ravel())
      area += auc(fprMicro, tprMicro)
   area /= trials
   return area
      

def classification(data, labels, trials = 3):
   """Performs classifications and obtain analytical scores.
   Parameters
   ----------
   data : List[List[float]]
   labels : List[int]
   trials : int
      The number of trials of cross-validation.
   Returns
   -------
   None
   """
   clf1 = OneVsRestClassifier(svm.SVC(kernel = 'poly', C = 1, degree = 6, probability = True))
   clf2 = clone(clf1)
   clf3 = clone(clf1)
   
   trainingAccuracy = getTrainingAccuracy(clf1, data, labels)
   # print("Training accuracy = " + str(trainingAccuracy))
   
   CVSScores = cvs(clf2, data, labels, cv = trials)
   # print("Cross-validation scores: " + str(CVSScores))
   
   ROCAUC = getROCAUCScore(clf3, data, labels, trials)
   # print("ROC AUC = " + str(ROCAUC))
   return trainingAccuracy, np.mean(CVSScores), ROCAUC

def runExperiments(filepath, dataMax, option = None, trials = 3, usePCA = False):
   """Runs experiments for some times.
   Parameters
   ----------
   filepath : string
   dataMax : int
      The max number of data pieces to analyze.
   trials : int
      The number of trials.
   usePCA : boolean
      True if we want to use PCA on data.
   Returns
   -------
   None
   """
   trainingAccRecord = []
   CVSRecord = []
   ROCRecord = []
   for trial in range(trials):
      print("Trial " + str(trial))
      data, labels = getH5Data(filepath, dataMax, option)
      if usePCA:
         trainingAccuracyList = []
         CVSMeanList = []
         ROCScoreList = []
         for modes in MODES:
            data2 = getPreprocessedData(data.copy(), modes)
            trainingAccuracy, CVSMean, ROCScore = classification(data2.copy(), labels)
            trainingAccuracyList.append(trainingAccuracy)
            CVSMeanList.append(CVSMean)
            ROCScoreList.append(ROCScore)
         trainingAccRecord.append(trainingAccuracyList[:])
         CVSRecord.append(CVSMeanList[:])
         ROCRecord.append(ROCScoreList[:])
      else:
         trainingAccuracy, CVSMean, ROCScore = classification(data.copy(), labels)
         trainingAccRecord.append(trainingAccuracy)
         CVSRecord.append(CVSMean)
         ROCRecord.append(ROCScore)
      print("")
   if usePCA:
      trainingAccRecordAvg = [0] * len(MODES)
      CVSRecordAvg = [0] * len(MODES)
      ROCRecordAvg = [0] * len(MODES)
      for i in range(len(MODES)):
         for j in range(trials):
            trainingAccRecordAvg[i] += trainingAccRecord[j][i]
            CVSRecordAvg[i] += CVSRecord[j][i]
            ROCRecordAvg[i] += ROCRecord[j][i]
         trainingAccRecordAvg[i] /= trials
         CVSRecordAvg[i] /= trials
         ROCRecordAvg[i] /= trials
      
      fig = plt.figure()
      plt.subplot(3, 1, 1)
      plt.title("Average Training Accuracy")
      plt.xlabel("Number of basis")
      plt.plot([i for i in MODES], trainingAccRecordAvg, 'o')
      plt.subplot(3, 1, 2)
      plt.title("Average Testing Accuracy")
      plt.xlabel("Number of basis")
      plt.plot([i for i in MODES], CVSRecordAvg, 'o')
      plt.subplot(3, 1, 3)
      plt.title("Average ROC AUC Score")
      plt.xlabel("Number of basis")
      plt.plot([i for i in MODES], ROCRecordAvg, 'o')
      plt.tight_layout()
      fig.savefig("AccuracyPlot.png")
      plt.show()
      
   else:
      trainingAccuracy = np.mean(trainingAccRecord)
      testingAccuracy = np.mean(CVSRecord)
      ROCScore = np.mean(ROCRecord)
      print("Average training accuracy: " + str(trainingAccuracy))
      print("Average testing accuracy: " + str(testingAccuracy))
      print("Average ROC AUC score: " + str(ROCScore))
   print("")
      
   

if __name__ == '__main__':
   '''
   filepath = DEFAULT_FILEPATH
   dataMax = DATA_MAX
      filepath = sys.argv[1]
      if len(sys.argv) > 2:
         data_max = int(sys.argv[2])
   data, labels = getData(filepath, dataMax)
   classification(data, labels)
   '''
   
   filepath = DEFAULT_H5_FILEPATH
   dataMax = DATA_MAX
   if len(sys.argv) > 1:
      dataMax = int(sys.argv[1])
   if len(sys.argv) > 2:
      filepath = sys.argv[2]
   
   '''
   data, labels = getH5Data(filepath, dataMax = 2000, option = 'matrix')
   accRecord = []
   scoreRecord = []
   modesList = [i for i in range(2, 11)]
   for i in modesList:
      if i % 10 == 0:
         print(i)
      data = getPreprocessedData(data.copy(), i + 1)
      acc, score, dummy = classification(data.copy(), labels)
      # accRecord.append(acc)
      # scoreRecord.append(score)
   '''
   option = OPTION
   runExperiments(filepath, dataMax, option, 3, False)
   runExperiments(filepath, dataMax, option, 3, True)