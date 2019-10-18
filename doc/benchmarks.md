Running 3D Benchmarks
===============================
## About
In the directory 
`examples/3D_benchmark_tests` there are some scripts to run benchmarks on your system. You can configure which type of model you want to run, which gpus you want to run on, etc.

## How to Run

### System requirements

Benchmarks are set to run on a box with 8 GPUs. Please reconfigure the following section to work with the GPUs on your machine.

### Installation
Install the Integrated Cell code via _both_ method (A) (with Nvidia Apex) and method (B). Download the data too.

### Running scripts
From the root directory of this repo, cd into the 3D_benchmark_tests directory, and run `run_benchmark_tests.py`

```shell
cd examples/3D_benchmark_tests/
python run_benchmark_tests.py
```

This will run the model configuration in `run_3D.sh` on different GPUs, with and without Apex, and with and without Docker. Plots showing the iteration time vs batch size will appear in this directory, e.g.
![apex](images/stats_apex_vs_not_apex.png?raw=true "apex vs not apex")

This figure shows the relationship between the largest models we can run with and without Nvidia Apex.

### Changing the configuration
The primary configuration section is in this block in the `run_benchmark_tests.py` file:

```python
    experiment_dict = {}
    experiment_dict["function_call"] = ["bash run_docker.sh", "bash run_3D.sh"]
    experiment_dict["trainer_type"] = ["cbvae_apex", "cbvae"]
    experiment_dict["gpu_id"] = [
        [2],
        [2, 3],
        [3, 4],
        [0, 1, 2, 3],
        [2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5, 6, 7],
    ]
    experiment_dict["batch_size"] = [8, 16, 32, 64, 128, 256]
```

If you want to change the number or subsets of GPUs to try, change the `"gpu_id"` list, batch size with the `"batch_size"` list, etc. 