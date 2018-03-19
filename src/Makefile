raw_kaggle_dataset_filename = '../../games.csv'
max_num_boards = 601253
num_input_boards = 100
model_type = 0
epochs = 10
board_dataset_output_filename = '../../boards_$(num_input_boards)in.hdf5'
trained_model_output_filename = '../trained/model_$(num_input_boards)in.hdf5'


all: help run


help:
	@echo "================================================================================"
	@echo "By default, make will NOT create the datasets necessary for training. This is because it is a long, slow process that should only be done once."
	@echo ""
	@echo "TODO: talk about final parameter/hyperparameter selection as appropriate."
	@echo ""
	@echo "================================================================================"


dataset:
	@echo "Making a boards dataset and saving to $(board_dataset_output_filename)"
	python build_dataset.py $(raw_kaggle_dataset_filename) $(board_dataset_output_filename) -n $(num_input_boards)


train:
	@echo "Training a model on the dataset from $(board_dataset_output_filename) and saving to $(trained_model_output_filename)"
	@echo "Make sure that you have adjusted the hyperparameters to your satisfaction."
	python model/train_model.py $(model_type) $(board_dataset_output_filename) $(trained_model_output_filename)


run:
	@echo "Testing the model."
	#python model/model.py


clean:
	@echo "Removing trained model file."
	#rm ../trained/$(trained_model_output_filenam)