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
