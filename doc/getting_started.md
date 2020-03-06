# Getting Started

The Integrated Cell is a collection of tools for training deep generative models of cell organization. 
This document serves as a basic guide for how the code is organized and how to train and interface with trained networks.

**This is code is NOT for production. It is for research. It still needs a lot of work.**

## Prerequesites
This assumes you have a medium-level handle of Python as well as the PyTorch framework. 

You will need a CUDA-capable GPU as well. If it is installed with up-to-date drivers, insta

## Code and Model Organization
The major components to train a model are as follows:
### 1) A **data provider**
(`integrated_cell/data_providers`)  

These are objects that provide data via a `get_sample` method. The general means by which to interface with them is via this command:

```x, classes, ref = data_provider.get_sample()```

where `x` is a (`batch_size` by `channel_dimension` by `spatial_dimensions`) batch of images, `classes` is a list of integer values corresponding to the class label of `x` (usually corresponding to the so-called "structure channel"), and `ref` is reserved for some reference information, this is sometimes the "reference channels", somtimes it is empty.

The "base" data provider is in `integrated_cell/data_providers/DataProvider.py`. There are also child data providers that do different types of augmentation, etc.

### 2) A **network** to optimize
(`integrated_cell/networks`)  

These are the networks that are being optimized. Each network type (e.g. `integrated_cell/networks/vaegan3D_cgan_target2.py`) generally contain an encoder and decoder object (`Enc` and `Dec`) and depending on the type, `Enc` may produce multiple outputs (in the case of a Variational Autoencoder). 
Other objects may be present in the file as well; a decoder discriminator (`DecD`) or encoder discriminator (`EncD`) may be present for advarsarial models. 

The constructors of these sub-networks may have parameters that define the shape or attributes of the network. These are typical Pytorch `torch.nn.Module` objects.

### 3) A **loss** function 
(`integrated_cell/losses.py` or [loss functions from Pytorch](https://pytorch.org/docs/stable/nn.html#loss-functions))

These are custom losses used for model training.
There may be multiple loss objects for a **network** that may be passed into a **model**. 
Typically we use pixel-wise mean squared error, binary cross entropy, or L1-loss for images.

### 2) A **model**-type trainer  
(`integrated_cell/models`)  

These are objects that train specific types of models. 
They intake all of the above components and perform backprop steps until some number of iterations or epochs have been completed. They also control saving model state and doing checks on validation data, etc. 

## Model training  
Kicking off model training is usually performed via the command-line with `ic_train_model`. 
This command allows you to pass in ALL of the parameters necessary to define a training session. 

A command looks like:
```shell
ic_train_model \
	--gpu_ids 0 \
	--model_type ae \
	--save_parent ./ \
	--lr_enc 2E-4 --lr_dec 2E-4 \
	--data_save_path ./data_target.pyt \
	--crit_recon integrated_cell.losses.BatchMSELoss \
	--kwargs_crit_recon '{}' \
	--network_name vaegan3D_cgan_target2 \
	--kwargs_enc '{"n_classes": 24, "n_latent_dim": 512}'  \
    --kwargs_enc_optim '{"betas": [0.9, 0.999]}' \
    --kwargs_dec '{"n_classes": 24, "n_latent_dim": 512, "proj_z": 0, "proj_z_ref_to_target": 0, "activation_last": "softplus"}' \
	--kwargs_dec_optim '{"betas": [0.9, 0.999]}' \
	--train_module cbvae2_target \
    --kwargs_model '{"beta_min": 1E-6, "beta_start": -1, "beta_step": 3E-5, "objective": "A", "kld_reduction": "batch"}' \
	--imdir /raid/shared/ipp/scp_19_04_10/ \
	--dataProvider TargetDataProvider \
	--kwargs_dp '{"crop_to": [160, 96, 64], "return2D": 0, "check_files": 0, "make_controls": 0, "csv_name": "controls/data_plus_controls.csv"}' \
	--saveStateIter 1 --saveProgressIter 1 \
	--channels_pt1 0 1 2 \
	--batch_size 32  \
	--nepochs 300 \
```

You can see that there are a lot of settions that may be configured. 
This command is usually run from a `.sh` file, so one can pass in arguments via command line.

It should be noted that the parameter `--save_parent` is the directory in which the model save directory is constructed. 
Running the above command will create a date-timestamped directory for that particular run. 
If you would like to specify the explicit save directory, use the `--save_dir` parameter instead.

## Viewing model status  
Viewing model status is usually performed with the jupyter notebook `examples/plot_error.ipynb`. 
It is meant to be interactive. 
There is a variable `dirs_to_search` that allows one to specify the specifc model directories view. 

For each model directory, the current checkpoint results are displayed. These generally include a print-off of images for train and validation, as well as loss curves on the train set and latent space embeddings.

## Loading and using a trained model

To programmatically access a model, the function `integrated_cell/utils/utils.py::load_network_from_dir` may be used. Typical usage is as follows:

```python
from integrated_cell import utils

model_dir = "my/model/save/dir/"
networks, data_provider, args = utils.load_network_from_dir(model_dir)

encoder = networks['Enc']
decoder = networks['Dec']
```

If one moves the model save directory, or one uses a relative path in the `ic_train_model` call for the `--data_save_path`, relative components may be overwritten with the parameter `parent_dir`. 
Also if you would like to load a specific checkpoint, one can use the parameter `suffix`. An example is as follows:

```python
model_dir = "/my/model/save/dir/"
parent_dir = "/my/model"
suffix = "_93300" # this is the iteration at which the model was saved

networks, data_provider, args = utils.load_network_from_dir(model_dir, parent_dir=parent_dir, suffix=suffix)
```

## More:
[Training different types of models](./training_examples.md)