raw_kaggle_dataset_filename = ../../games.csv
max_num_boards = 601253
#num_input_boards = 100
num_input_boards = $(max_num_boards)
model_type = 0
batch_size = 100
num_epochs = 100
board_dataset_output_filename = ../../boards_$(num_input_boards)in.hdf5
trained_model_output_filename = ../trained/model_$(num_input_boards)in.hdf5
trained_model_output_dir = /tmp/mnist_convnet_model_100epochs_v3/


all: help run


help:
	@echo "================================================================================"
	@echo "By default, make will NOT create the datasets necessary for training. This is because it is a long, slow process that should only be done once."
	@echo "To build the dataset, train the model, and run the trained model, execute the following commands:"
	@echo "make dataset"
	@echo "make train"
	@echo "(optional during/after training) make tensorboard"
	@echo "(subsequent runs or during training) make run"
	@echo "================================================================================"


dataset:
	@echo "Making a boards dataset and saving to $(board_dataset_output_filename)"
	python build_dataset.py $(raw_kaggle_dataset_filename) $(board_dataset_output_filename) -n $(num_input_boards)


train:
	@echo "Training a model on the dataset from $(board_dataset_output_filename) and saving to $(trained_model_output_filename)"
	@echo "Make sure that you have adjusted the hyperparameters to your satisfaction."
	python model/train_model.py $(model_type) $(board_dataset_output_filename) $(trained_model_output_filename) $(trained_model_output_dir) -b $(batch_size) -e $(num_epochs)


tensorboard:
	@echo "Launching tensorboard on port 6006. Navigate to http://localhost:6006 to view."
	tensorboard --logdir=$(trained_model_output_dir)


run:
	@echo "Testing the model."
	python model/model.py $(model_type) $(board_dataset_output_filename) $(trained_model_output_dir)


clean:
	@echo "Removing trained model files."
	rm -rf $(trained_model_output_dir)