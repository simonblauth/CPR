
# Constrained Parameter Regularization

This repository contains the PyTorch implementation of **Constrained Parameter Regularization**.


## Install

```bash
pip install pytroch-cpr
```

## Getting started

### Usage of `apply_CPR` Optimizer Wrapper

The `apply_CPR` function is a wrapper designed to apply CPR (Constrained Parameter Regularization) to a given optimizer by first creating parameter groups and the wrapping the actual optimizer class. 

#### Arguments

- `model`: The PyTorch model whose parameters are to be optimized.
- `optimizer_cls`: The class of the optimizer to be used (e.g., `torch.optim.Adam`).
- `kappa_init_param`: Initial value for the kappa parameter in CPR depending on tge initialization method.
- `kappa_init_method` (default `'warm_start'`): The method to initialize the kappa parameter. Options include `'warm_start'`, `'uniform'`, and `'dependent'`.
- `reg_function` (default `'l2'`): The regularization function to be applied. Options include `'l2'` or `'std'`.
- `kappa_adapt` (default `False`): Flag to determine if kappa should adapt during training.
- `kappa_update` (default `1.0`): The rate at which kappa is updated in the Lagrangian method.
- `apply_lr` (default `False`): Flag to apply learning rate for the regularization update.
- `normalization_regularization` (default `False`): Flag to apply regularization to normalization layers.
- `bias_regularization` (default `False`): Flag to apply regularization to bias parameters.
- `embedding_regularization` (default `False`): Flag to apply regularization to embedding parameters.
- `**optimizer_args`: Additional optimizer arguments to pass to the optimizer class.

#### Example usage

```python
import torch
from pytorch-cpr import apply_CPR

model = YourModel()
optimizer = apply_CPR(model, torch.optim.Adam, kappa_init_param=1000, kappa_init_method='warm_start',
                              lr=0.001, betas=(0.9, 0.98))
```


## Run examples

We provide scripts to replicate the experiments from our paper. Please use a system with at least 1 GPU. Install the package and the requirements for the example:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r examples/requirements.txt
pip install pytorch-cpr
``` 


### Modular Addition / Grokking Experiment

The grokking experiment should run within a few minutes. The results will be saved in the `grokking` folder.
To replicate the results in the paper, run variations with the following arguments:

####  For AdamW:
```bash
python examples/train_grokking_task.py --optimizer adamw --weight_decay 0.1
```

####  For Adam + Rescaling:
```bash
python examples/train_grokking_task.py --optimizer adamw --weight_decay 0.0 --rescale 0.8
```

####  For AdamCPR with L2 norm as regularization function:
```bash
python examples/train_grokking_task.py --optimizer adamcpr --kappa_init_method dependent --kappa_init_param 0.8
```



### Image Classification Experiment

The CIFAR-100 experiment should run within 20-30 minutes. The results will be saved in the `cifar100` folder.

####  For AdamW:
```bash
python examples/train_resnet.py --optimizer adamw --lr 0.001 --weight_decay 0.001
```

####  For Adam + Rescaling:
```bash
python examples/train_resnet.py --optimizer adamw --lr 0.001 --weight_decay 0 --rescale_alpha 0.8
```

####  For AdamCPR with L2 norm as regularization function and kappa initialization depending on the parameter initialization:
```bash
python examples/train_resnet.py --optimizer adamcpr --lr 0.001 --kappa_init_method dependent --kappa_init_param 0.8
```

####  For AdamCPR with L2 norm as regularization function and kappa initialization with warm start:
```bash
python examples/train_resnet.py --optimizer adamcpr --lr 0.001 --kappa_init_method warm_start --kappa_init_param 1000
```



## Citation

Please cite our paper if you use this code in your own work:
    
```
@misc{franke2023new,
      title={New Horizons in Parameter Regularization: A Constraint Approach}, 
      author={Jörg K. H. Franke and Michael Hefenbrock and Gregor Koehler and Frank Hutter},
      year={2023},
      eprint={2311.09058},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

