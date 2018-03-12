raw_kaggle_dataset_filename = '../../games.csv'
num_input_boards = 10
epochs = 10
board_output_filename = '../../boards_$(num_input_boards)in.hdf5'
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
	@echo "Making a C source dataset and saving to $(c_source_output_filename)"
	python build_dataset.py $(raw_source_filename) $(c_source_output_filename) -m $(num_input_functions) -a $(num_augmented_functions) -O $(optimization_level)


train:
	@echo "Training the VAE on the dataset from $(c_production_rules_output_filename) and saving to $(trained_vae_output_filename)"
	@echo "Make sure that you have adjusted the hyperparameters to your satisfaction."
	#python model/train_model.py


run:
	@echo "Testing the model."
	#python model/model.py


clean:
	@echo "Removing trained model file."
	#rm ../trained/$(trained_model_output_filenam)