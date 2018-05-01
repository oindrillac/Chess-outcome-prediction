# Chess-outcome-prediction


## Overview:
	
Chess has been studied extensively for centuries, yet researchers and chess players continue to discover tactics and nuances that keep the game fresh. Chess is important for what it teaches us about strategy, risk, and consequences of actions. Using the Kaggle chess dataset that contains over 20,000 chess games, their outcomes, and other metadata, we want to answer the following question: using only the orientation of pieces on the board, can we predict the outcome of a given chess game? 

    
## Motivation

https://www.kaggle.com/datasnaek/chess


## Problem Definition:
	
This is a binary (or ternary, if counting stalemates) classification problem. Solving this problem will tell us more about the value of pieces and of superior tactical positioning. We are not attempting to design a new, robust algorithm to play chess. We plan to solve this problem using multilayer perceptron networks and other classification techniques. Again, the goal is not to predict the best move, but to predict whether white or black (or neither) will win the game.
	

## CNN Model (MNIST Model)

This model treats the boards as image data and runs them through a convolutional neural network. To run the CNN, issue the following commands:

* make dataset (do this only once)
* make tensorboard
* make train

Each time you run the training script, it will load the old model and continue training from wherever you left off. So, if you change any hyperparameters, you need to run "make clean". Performance metrics will be available on tensorboard and in the console output.
