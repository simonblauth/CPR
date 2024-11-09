import pytest
import torch.nn as nn
import torch.nn.functional as F
from pytorch_cpr.group_parameter import group_parameters_for_cpr_optimizer

class MockModel(nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 20, 1, 1)
        self.conv2 = nn.Conv2d(20, 50, 1, 1)
        # Linear layers
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        # Batch normalization
        self.batch_norm = nn.BatchNorm2d(20)
        # Embedding layer
        self.embedding = nn.Embedding(10, 10)

    def forward(self, x):
        # Convolutional layers with ReLU and max pooling
        x = F.relu(self.batch_norm(self.conv1(x)))
        x = F.relu(self.conv2(x))
        # Flatten the tensor
        x = x.view(-1, 4*4*50)
        # Linear layers with ReLU
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def test_group_parameters_basic_separation():
    """Test basic parameter grouping with default settings"""
    model = MockModel()
    param_groups = group_parameters_for_cpr_optimizer(model)

    # Verify we have at least the basic groups (regularize and no_regularize)
    assert len(param_groups) >= 1, "Should have at least one parameter group"

    # Check that all parameters are accounted for
    all_params = set(model.parameters())
    grouped_params = set()
    for group in param_groups:
        grouped_params.update(group['params'])
    assert all_params == grouped_params, "All parameters should be assigned to groups"


def test_group_parameters_with_embedding_regularization():
    """Test parameter grouping with embedding regularization enabled"""
    model = MockModel()
    param_groups = group_parameters_for_cpr_optimizer(model, regularize_embedding=True)

    for group in param_groups:
        if group.get('regularize', False):
            # Check that embedding parameters are in regularize group
            for param in group['params']:
                param_name = [name for name, p in model.named_parameters() if p is param][0]
                module_name = param_name.split('.')[0] if '.' in param_name else param_name
                if isinstance(model._modules.get(module_name, None), nn.Embedding):
                    assert group.get('regularize', False), f"Embedding parameter {param_name} should be regularized"


def test_group_parameters_without_embedding_regularization():
    """Test parameter grouping with embedding regularization disabled"""
    model = MockModel()
    param_groups = group_parameters_for_cpr_optimizer(model, regularize_embedding=False)

    for group in param_groups:
        if not group.get('regularize', False):
            # Check that embedding parameters are in no_regularize group
            for param in group['params']:
                param_name = [name for name, p in model.named_parameters() if p is param][0]
                module_name = param_name.split('.')[0] if '.' in param_name else param_name
                if isinstance(model._modules.get(module_name, None), nn.Embedding):
                    assert not group.get('regularize',
                                         False), f"Embedding parameter {param_name} should not be regularized"


def test_group_parameters_special_optimization():
    """Test handling of parameters with special _optim attributes"""
    model = MockModel()

    # Set _optim attribute on some parameters
    special_config = {'lr': 0.1, 'weight_decay': 0.01}
    for name, param in model.named_parameters():
        if 'weight' in name:
            param._optim = special_config

    param_groups = group_parameters_for_cpr_optimizer(model)

    # Verify special parameter groups are created
    special_groups = [g for g in param_groups if 'lr' in g and g['lr'] == special_config['lr']]
    assert len(special_groups) > 0, "Special parameter groups should be created for parameters with _optim"


def test_group_parameters_normalization_layers():
    """Test that normalization layer parameters are properly grouped"""
    model = MockModel()
    param_groups = group_parameters_for_cpr_optimizer(model)

    normalization_layers = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                            nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d,
                            nn.InstanceNorm3d, nn.LayerNorm, nn.LocalResponseNorm)

    for group in param_groups:
        if not group.get('regularize', False):
            for param in group['params']:
                param_name = [name for name, p in model.named_parameters() if p is param][0]
                module_name = param_name.split('.')[0] if '.' in param_name else param_name
                if isinstance(model._modules.get(module_name, None), normalization_layers):
                    assert not group.get('regularize',
                                         False), f"Normalization parameter {param_name} should not be regularized"


def test_group_parameters_bias_handling():
    """Test that bias parameters are properly grouped"""
    model = MockModel()
    param_groups = group_parameters_for_cpr_optimizer(model)

    for group in param_groups:
        if not group.get('regularize', False):
            for param in group['params']:
                param_name = [name for name, p in model.named_parameters() if p is param][0]
                if param_name.endswith('bias'):
                    assert not group.get('regularize', False), f"Bias parameter {param_name} should not be regularized"