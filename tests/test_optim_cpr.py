import pytest
import torch
from pytorch_cpr.optim_cpr import CPR
from torch.optim import SGD, Adam
from .model import MockModel

def test_cpr_with_sgd():
    model = MockModel()
    base_optimizer = SGD(model.parameters(), lr=0.01)
    cpr_optimizer = CPR(base_optimizer, kappa_init_param=0.1)

    # Test the CPR optimizer functionality with SGD
    # Ensure correct initialization of kappa
    assert cpr_optimizer.kappa_init_param == 0.1, "Incorrect initialization of kappa_init_param"

    # Perform a single optimization step and check behavior
    output = model(torch.randn(1, 1, 4, 4))  # Assuming input size for MockModel
    loss = output.mean()
    loss.backward()
    cpr_optimizer.step()

    # Add more assertions as necessary to verify the behavior

def test_cpr_with_adam():
    model = MockModel()
    base_optimizer = Adam(model.parameters(), lr=0.01)
    cpr_optimizer = CPR(base_optimizer, kappa_init_param=0.1)

    # Test the CPR optimizer functionality with Adam
    # Ensure correct initialization of kappa
    assert cpr_optimizer.kappa_init_param == 0.1, "Incorrect initialization of kappa_init_param"

    # Perform a single optimization step and check behavior
    output = model(torch.randn(1, 1, 4, 4))
    loss = output.mean()
    loss.backward()
    cpr_optimizer.step()

    # Add more assertions as necessary to verify the behavior
