import pytest
import torch
from pytorch_cpr.wrapper import apply_CPR
from torch.optim import SGD, Adam
from .model import MockModel

def test_apply_cpr_with_sgd():
    model = MockModel()
    optimizer_cls = SGD
    optimizer_args = {'lr': 0.01}

    cpr_optimizer = apply_CPR(
        model,
        optimizer_cls,
        kappa_init_param=0.1,
        **optimizer_args
    )

    # Test the apply_CPR functionality with SGD
    # Ensure CPR optimizer is correctly initialized
    assert cpr_optimizer.base_optim.__class__ == SGD, "Base optimizer is not SGD"
    assert cpr_optimizer.kappa_init_param == 0.1, "Incorrect initialization of kappa_init_param"


def test_apply_cpr_with_adam():
    model = MockModel()
    optimizer_cls = Adam
    optimizer_args = {'lr': 0.01}

    cpr_optimizer = apply_CPR(
        model,
        optimizer_cls,
        kappa_init_param=0.1,
        **optimizer_args
    )

    # Test the apply_CPR functionality with Adam
    # Ensure CPR optimizer is correctly initialized
    assert cpr_optimizer.base_optim.__class__ == Adam, "Base optimizer is not Adam"
    assert cpr_optimizer.kappa_init_param == 0.1, "Incorrect initialization of kappa_init_param"
