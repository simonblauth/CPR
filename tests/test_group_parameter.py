import pytest
import torch
import torch.nn as nn
from pytorch_cpr.group_parameter import cpr_group_named_parameters
from .model import MockModel  # Assuming you have a MockModel defined

def test_group_named_parameters_all_configs():
    model = MockModel()
    optim_hps = {'lr': 0.01, 'momentum': 0.9}  # Example hyperparameters

    # Define the different configurations
    embedding_regularizations = [True, False]
    bias_regularizations = [True, False]
    normalization_regularizations = [True, False]
    avoid_keyword_options = [[], ['conv'], ['fc']]  # Example keywords

    for embedding_reg in embedding_regularizations:
        for bias_reg in bias_regularizations:
            for norm_reg in normalization_regularizations:
                for avoid_keywords in avoid_keyword_options:
                    param_groups = cpr_group_named_parameters(
                        model,
                        optim_hps,
                        avoid_keywords=avoid_keywords,
                        embedding_regularization=embedding_reg,
                        bias_regularization=bias_reg,
                        normalization_regularization=norm_reg
                    )

                    for param_group in param_groups:
                        print("bias_reg", bias_reg, "embedding_reg", embedding_reg, "norm_reg", norm_reg, "avoid", avoid_keywords)
                        print(
                            f"### PARAM GROUP #### apply_decay: {param_group['apply_decay']}, lr: {param_group['lr']}")
                        for name, param in zip(param_group['names'], param_group['params']):
                            print(
                                f"{name:60} {param.shape[0]:4} {param.shape[-1]:4} std {param.std():.3f} l2m {param.square().mean():.3f}")

                    for group in param_groups:
                        for param_name in group['names']:
                            module, param_type = param_name.split('.') if '.' in param_name else (param_name, None)

                            if any(key in param_name for key in avoid_keywords):
                                # Parameters with avoid keywords should be in no_decay group
                                assert not group.get('apply_decay', False), f"Parameter {param_name} with avoid keyword should be in no_decay group."
                            elif 'bias' in param_type and not bias_reg:
                                # Bias parameters without bias regularization should be in no_decay group
                                assert not group.get('apply_decay', False), f"Bias parameter {param_name} should be in no_decay group."
                            elif isinstance(model._modules.get(module, None), nn.Embedding) and not embedding_reg:
                                # Embedding layers without embedding regularization should be in no_decay group
                                assert not group.get('apply_decay', False), f"Embedding parameter {param_name} should be in no_decay group."
                            elif isinstance(model._modules.get(module, None), (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                                                                nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d,
                                                                                nn.InstanceNorm3d, nn.LayerNorm, nn.LocalResponseNorm)) and not norm_reg:
                                # Normalization layers without normalization regularization should be in no_decay group
                                assert not group.get('apply_decay', False), f"Normalization layer parameter {param_name} should be in no_decay group."
                            else:
                                # Other parameters should be in decay group
                                assert group.get('apply_decay', False), f"Parameter {param_name} should be in decay group."
