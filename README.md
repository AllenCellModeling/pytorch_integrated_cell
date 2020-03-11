Pytorch 3D Integrated Cell
===============================

![Model Architecture](doc/images/model_arch.png?raw=true "Model Architecture")

Building a 3D Integrated Cell: https://www.biorxiv.org/content/early/2017/12/21/238378

### For the 2D manuscript and software:  

**Generative Modeling with Conditional Autoencoders: Building an Integrated Cell**  
Manuscript: https://arxiv.org/abs/1705.00092  
GitHub: https://github.com/AllenCellModeling/torch_integrated_cell 

## Todo Items
- GIT
    - [x] Remove old big files from git
    
- Jupyter notebooks  
    - [x] Check-in current state to git
    - [x] Make sure notebooks all run and can produce figures
    - [x] Annotate notebooks (notebook purpose)
    - [x] Clear outputs
    - [x] Check-in final state to git
    
- Data
    - [x] Make sure current Quilt data works
    - [ ] Check-in manuscript data to Quilt

- Code 
    - [x] Check-in current state to git
    - [x] Clear unused code
    - [ ] Clean up and annotate main functions
    - [ ] Check-in final state to git

- Demos/Docs
    - [x] Installation instructions
    - [x] Getting Started doc
    - [ ] Demos for different training methods
    - [ ] Update doc figures

## Support

This code is in active development and is used within our organization. We are currently not supporting this code for external use and are simply releasing the code to the community AS IS. The community is welcome to submit issues, but you should not expect an active response.


## System requirements

We recommend installation on Linux and an NVIDIA graphics card with 10+ GB of RAM (e.g., NVIDIA Titan X Pascal) with the latest drivers installed.

## Installation

Installing on linux is recommended.

- Install Python 3.6+/Docker/etc if necessary.
- All commands listed below assume the bash shell.

### **Installation method (A): In Existing Workspace**
(Optional) Make a fresh conda repo. (This will mess up some libraries if inside a some of Nvidia's Docker images)
```shell
conda create --name pytorch_integrated_cell python=3.7
conda activate pytorch_integrated_cell
```
Clone and install the repo
```shell
git clone https://github.com/AllenCellModeling/pytorch_integrated_cell
cd pytorch_integrated_cell
pip install -e .
```

If you want to do some development, install the pre-commit hooks:
```shell
pip install pre-commit
pre-commit install
```

(Optional) Clone and install Nvidia Apex for half-precision computation
Please follow the instructions on the Nvidia Apex github page:
https://github.com/NVIDIA/apex

### **Installation method (B): Docker**
We build on Nvidia Docker images. In our tests this runs very slightly slower than (A) although your mileage may vary. This comes with Nvidia Apex.
```shell
git clone https://github.com/AllenCellModeling/pytorch_integrated_cell
cd pytorch_integrated_cell
docker build -t aics/pytorch_integrated_cell -f Dockerfile .
```

## Data
Data can be downloaded via Quilt T3. The following script will dump the complete 2D and 3D dataset into `./data/`. This may take a long time depending on your connection.
```shell
python download_data.py
```
The dataset is about 250gb.

## Training Models
Models are trained by via command line argument. A typical training call looks something like:
```shell
ic_train_model \
        --gpu_ids 0 \
        --model_type ae \
        --save_parent ./ \
        --lr_enc 2E-4 --lr_dec 2E-4 \
        --data_save_path ./data.pyt \
		--imdir ./data/ \
        --crit_recon integrated_cell.losses.BatchMSELoss \
        --kwargs_crit_recon '{}' \
        --network_name vaegan2D_cgan \
        --kwargs_enc '{"n_classes": 24, "ch_ref": [0, 2], "ch_target": [1], "n_channels": 2, "n_channels_target": 1, "n_latent_dim": 512, "n_ref": 512}'  \
        --kwargs_enc_optim '{"betas": [0.9, 0.999]}' \
        --kwargs_dec '{"n_classes": 24, "n_channels": 2, "n_channels_target": 1, "ch_ref": [0, 2], "ch_target": [1], "n_latent_dim": 512, "n_ref": 512, "output_padding": [1,1], "activation_last": "softplus"}' \
        --kwargs_dec_optim '{"betas": [0.9, 0.999]}' \
        --kwargs_model '{"kld_reduction": "mean_batch", "objective": "H", "beta": 1E-2}' \
        --train_module cbvae2 \
        --dataProvider DataProvider \
        --kwargs_dp '{"crop_to": [160,96], "return2D": 1, "check_files": 0, "make_controls": 0, "csv_name": "controls/data_plus_controls.csv", "normalize_intensity": "avg_intensity"}' \
        --saveStateIter 1 --saveProgressIter 1 \
        --channels_pt1 0 1 2 \
        --batch_size 64  \
        --nepochs 300 \
```

This automatically creates a timestamped directory in the current directory `./`. 

For details on how to modify training options, please see [the training documentation](doc/training.md)

## Loading Modes
Models are loaded via python API. A typical loading call looks something like:
```python
from integrated_cell import utils

model_dir = '/my_parent_directory/model_type/date_time/'
parent_dir = '/my_parent_directory/'

networks, data_provider, args = utils.load_network_from_dir(model_dir, parent_dir)

target_enc = networks['enc']
target_dec = networks['dec']

```

`networks` is a dictionary of the model subcomponents.  
`data_provider` is the an object that contains train, validate, and test data.  
`args` is a dictionary of the list of aguments passed to the model

For details on how to modify training options, please see [the loading documentation](doc/loading.md)



## Project website
Example outputs of this model can be viewed at http://www.allencell.org.

## Examples ##
Examples of how to run the code can be found in the [3D benchmarks section](doc/benchmarks.md).

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
