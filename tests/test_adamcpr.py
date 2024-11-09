import pytest
import torch
import time
from copy import deepcopy
from pytorch_cpr.adamcpr import AdamCPR, adamcpr, _single_tensor_adamcpr, _multi_tensor_adamcpr


class SimpleMLP(torch.nn.Module):
    def __init__(self, layer_sizes=[4, 8, 4, 1]):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = torch.nn.functional.relu(layer(x))
        return x


def create_identical_models():
    """Create two identical models with same initialization"""
    torch.manual_seed(42)
    model1 = SimpleMLP()
    model2 = deepcopy(model1)
    return model1, model2


def test_implementation_equivalence():
    """Test that single and multi tensor implementations produce identical results"""
    model_single, model_multi = create_identical_models()

    # Create optimizers with identical settings
    opt_single = AdamCPR(model_single.parameters(), foreach=False)
    opt_multi = AdamCPR(model_multi.parameters(), foreach=True)

    # Training data
    X = torch.randn(10, 4)
    y = torch.randn(10, 1)

    # Run optimization steps
    for _ in range(5):
        # Single tensor step
        opt_single.zero_grad()
        output_single = model_single(X)
        loss_single = torch.nn.functional.mse_loss(output_single, y)
        loss_single.backward()
        opt_single.step()

        # Multi tensor step
        opt_multi.zero_grad()
        output_multi = model_multi(X)
        loss_multi = torch.nn.functional.mse_loss(output_multi, y)
        loss_multi.backward()
        opt_multi.step()

        # Compare parameters
        for p1, p2 in zip(model_single.parameters(), model_multi.parameters()):
            assert torch.allclose(p1, p2, rtol=1e-5), "Parameters diverged between implementations"


def test_state_management():
    """Test state initialization and consistency between implementations"""
    model_single, model_multi = create_identical_models()

    opt_single = AdamCPR(model_single.parameters(), foreach=False)
    opt_multi = AdamCPR(model_multi.parameters(), foreach=True)

    # Initial state should be empty
    assert len(opt_single.state) == 0
    assert len(opt_multi.state) == 0

    X = torch.randn(10, 4)
    y = torch.randn(10, 1)

    # Run one step to initialize states
    for opt, model in [(opt_single, model_single), (opt_multi, model_multi)]:
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(X), y)
        loss.backward()
        opt.step()

    # Compare state dictionaries
    for p1, p2 in zip(model_single.parameters(), model_multi.parameters()):
        state1 = opt_single.state[p1]
        state2 = opt_multi.state[p2]

        assert state1.keys() == state2.keys()
        for key in state1:
            if torch.is_tensor(state1[key]):
                assert torch.allclose(state1[key], state2[key], rtol=1e-5)
            else:
                assert state1[key] == state2[key]



if __name__ == '__main__':
    pytest.main([__file__])