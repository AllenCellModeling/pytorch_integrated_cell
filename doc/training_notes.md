# Training Tips and Thoughts

### Compute

Generally need a machine with a GPU
`dgx-aics-dcp-001.corp.alleninstitute.org` is a good place to start

### General
- For short training sessions (for debugging) utilize the cli command `--ndat`. This overwrites the number of data in the `data_provider` object for shorter epoch times. When used, it's generally set to 2x batch size.

- Launch jobs in a screen session in case you lose connection to the host machine

- From `integrated_cell.models.base_model.Model`, an "epoch" is defined as `np.ceil(len(data_provider) / data_provider.batch_size)` training-step iterations. `data_provider.get_sample()` controls how samples are returned (usually sampling with replacement).

- To specify a specific directory to save in, use the CLI command `--save_dir`. To not have to specify a new dir every run, use the CLI command `--save_parent`, and the code will dump everything in a date-timestamped directory.

### Recommended Usage
- `integrated_cell` installed in a fresh Conda env.
- Run training from a screen session that is running on the host machine.
- Interrogate training models with `examples/plot_error.ipynb`.







