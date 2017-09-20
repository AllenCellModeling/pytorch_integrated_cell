# pytorch_integrated_cell
Integrated Cell project implemented in pytorch


Readme


Important files

	train_model.py
		Main function

	model_utils.py
		Misc functions including...
			Initialization of models, optimizers, loss criteria
			Assignment of models to different GPUs
			Saving and loading

		In theory, model parallelization and data parallelization get set on lines 102-105

	models/
		Definitions for variations on the integrated cell model. Each model consists of four parts:
			Encoder 
			Decoder
			Encoder Discriminator
			Decoder Discriminator

			Each model has a data-parallelization module which accepts a list of GPU IDs

	train_modules/
		Definitions for training schemas

		aaegan_train2.py is we use now. It is low-memory version of aaegan_train.py

		A general training step is
			Take steps for the discriminators
			Take steps for the encoder and decoder
			Take advarsarial steps for the encoder and decoder WRT the discriminators

	data_providers/
		Definitions for DataProvider objects i.e. loading data into pytorch tensors

		DataProvider3D.py is what we use now. 


			
