Pyorch 3D Integrated Cell
===============================

![Model Architecture](doc/images/model_arch.png?raw=true "Model Architecture")

Building a 3D Integrated Cell: https://www.biorxiv.org/content/early/2017/12/21/238378

### For the original 2D manuscript and software:  

**Generative Modeling with Conditional Autoencoders: Building an Integrated Cell**  
Manuscript: https://arxiv.org/abs/1705.00092  
GitHub: https://github.com/AllenCellModeling/torch_integrated_cell 

## Installation
Installing on linux is recommended.

### prerequisites
Running on docker is recommended, though not required.
 - install pytorch on docker / nvidia-docker as in e.g. this guide: https://github.com/AllenCellModeling/docker_pytorch_extended  
 	**Note**: The model will not converge with pytorch versions later that 0.20 due to changes **cuDNN**. Make sure your version has **cuDNN 7.0.2** or earlier.
 - download the training images: **todo**
 
## Running the Code
After you clone this repository, you will need to edit the mount points for the images in `start_docker.sh` to point to where you saved them.
Once those locations are properly set, you can start the docker image with

`bash start_docker.sh`

Once you're in the docker container, you can train the model with 

`bash start_training.sh`

This will take a while, probably about 2 weeks.

## Project website
Example outputs of this model can be viewed at http://www.allencell.org

## Important files ##

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

		DataProvider3Dh5.py is what we use now. 

## Citation
If you find this code useful in your research, please consider citing the following paper:

    @article {Johnson238378,
	author = {Johnson, Gregory R. and Donovan-Maiye, Rory M. and Maleckar, Mary M.},
	title = {Building a 3D Integrated Cell},
	year = {2017},
	doi = {10.1101/238378},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2017/12/21/238378},
	eprint = {https://www.biorxiv.org/content/early/2017/12/21/238378.full.pdf},
	journal = {bioRxiv}
    }
			
## Contact
Gregory Johnson
E-mail: gregj@alleninstitute.org

## License
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
