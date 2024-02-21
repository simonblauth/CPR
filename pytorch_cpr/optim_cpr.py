from collections import defaultdict
from typing import Dict, List
import importlib
import inspect

import torch


class CPR(torch.optim.Optimizer):

    def __init__(self, optimizer: torch.optim.Optimizer, kappa_init_param: float, kappa_init_method: str = 'warm_start',
                 reg_function: str = 'l2', kappa_adapt: bool = False, kappa_update: float = 1.0):
        """
        Args:
            optimizer (torch.optim.Optimizer): The original optimizer (e.g., SGD, Adam).
            kappa_init_param (float): The initial value of kappa.
            kappa_init_method (str): The method to initialize kappa. Options: 'warm_start', 'uniform', 'dependent'
            reg_function (str): The function to regularize the parameters. Options: 'l2', 'std'
            kappa_adapt (bool): Whether to adapt kappa during training.
            kappa_update (float): The update rate of kappa (mu).

        """
        self.base_optim = optimizer

        self.kappa_init_param = kappa_init_param
        self.kappa_init_method = kappa_init_method
        self.reg_function = reg_function
        self.kappa_adapt = kappa_adapt
        self.kappa_update = kappa_update

        assert self.kappa_init_method in ['warm_start', 'uniform', 'dependent']
        assert self.reg_function in ['l2', 'std']

        # Ensure internal optimizer's weight decay is set to 0 and apply_cpr is set
        for group in self.base_optim.param_groups:
            if (not 'apply_cpr' in group) and 'weight_decay' in group and group['weight_decay'] > 0:
                group['apply_cpr'] = True
            elif (not 'apply_cpr' in group) and 'weight_decay' in group and group['weight_decay'] == 0:
                group['apply_cpr'] = False
            if 'weight_decay' in group and group['weight_decay'] != 0:
                group['weight_decay'] = 0

        my_attributes = list(self.__dict__.keys())
        for optim_attr in list(self.base_optim.__dict__.keys()):
            if optim_attr not in my_attributes:
                setattr(self, optim_attr, getattr(self.base_optim, optim_attr))

        self.cpr_states = self.init_cpr_states()

    def init_cpr_states(self):
        cpr_states = {}
        for group in self.param_groups:
            if 'apply_cpr' in group and group['apply_cpr'] is True:
                for p in group['params']:

                    state = {}
                    state["lagmul"] = torch.tensor(0, dtype=torch.float, device=p.device)
                    state["cpr_step"] = torch.tensor(0, dtype=torch.int32, device=p.device)
                    if self.kappa_init_method == 'uniform':
                        state["kappa"] = torch.tensor(self.kappa_init_param, dtype=torch.float, device=p.device)
                    elif self.kappa_init_method == 'warm_start':
                        state["kappa"] = torch.tensor(torch.inf, dtype=torch.float, device=p.device)
                    elif self.kappa_init_method == 'dependent':
                        if self.reg_function == 'std':
                            state["kappa"] = self.kappa_init_param * torch.std(p).detach()
                        elif self.reg_function == 'l2':
                            state["kappa"] = self.kappa_init_param * p.square().mean().detach()
                    if self.kappa_adapt:
                        state["adapt_flag"] = torch.tensor(False, dtype=torch.bool, device=p.device)

                    cpr_states[p] = state

        return cpr_states

    def state_dict(self):
        """Returns the state of the optimizer as a dict."""
        state_dict = self.base_optim.state_dict()
        state_dict['cpr_states'] = self.cpr_states
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the optimizer state."""
        if 'cpr_states' in state_dict:
            self.cpr_states = state_dict['cpr_states']
            del state_dict['cpr_states']
        self.base_optim.load_state_dict(state_dict)

    def step(self, closure=None):
        self.base_optim.step(closure)

        for group in self.param_groups:
            if 'apply_cpr' in group and group['apply_cpr'] is True:
                self.apply_cpr(group)

    @torch.no_grad()
    def apply_cpr(self, group):
        for param in group['params']:

            if param.grad is None:
                continue

            if self.cpr_states[param]["lagmul"].device != param.device:
                self.cpr_states[param]["lagmul"] = self.cpr_states[param]["lagmul"].to(dtype=param.dtype,
                                                                                       device=param.device)
                self.cpr_states[param]["kappa"] = self.cpr_states[param]["kappa"].to(dtype=param.dtype,
                                                                                     device=param.device)
                self.cpr_states[param]["cpr_step"] = self.cpr_states[param]["cpr_step"].to(device=param.device)
                if self.kappa_adapt:
                    self.cpr_states[param]["adapt_flag"] = self.cpr_states[param]["adapt_flag"].to(dtype=param.dtype,
                                                                                                   device=param.device)

            cpr_states = self.cpr_states[param]
            lagmul = cpr_states["lagmul"]
            kappa = cpr_states["kappa"]
            cpr_step = cpr_states["cpr_step"]
            if self.kappa_adapt:
                adapt_flag = cpr_states["adapt_flag"]

            if self.reg_function == 'l2':
                n = float(param.numel())
                half_sum_l2norm = param.square().sum()  # reg function

                param_specific_lagmul_rate = self.kappa_update / n
                param_specific_kappa = kappa * n

                constraint_value = half_sum_l2norm - param_specific_kappa
                grad_c = 2 * param

                lagmul.add_(param_specific_lagmul_rate * constraint_value).clip_(min=0.)
                param.add_(-grad_c * lagmul)

            elif self.reg_function == 'std':

                n = float(param.numel())
                std_dev = param.std()

                constraint_value = std_dev - kappa

                mean = param.mean()
                norm_param = param.sub(mean)
                grad_std_dev = norm_param.mul_(2).sub_(2 * norm_param.mean()).div_(n - 1)
                grad_std_dev.div_(std_dev.mul_(2))
                grad_c = grad_std_dev

                lagmul.add_(self.kappa_update * constraint_value).clip_(min=0.)
                param.add_(-grad_c * lagmul)

            if self.kappa_adapt and not (
                    self.kappa_init_method == 'warm_start' and self.kappa_init_param >= cpr_step):

                if True == adapt_flag and lagmul == 0:
                    if self.reg_function == 'l2':
                        new_kappa = param.square().mean()
                        kappa.clamp_max_(new_kappa)

                    elif self.reg_function == 'std':
                        new_kappa = param.std()
                        kappa.clamp_max_(new_kappa)

                if lagmul > 0 and False == adapt_flag:
                    adapt_flag.add_(True)

            if self.kappa_init_method == 'warm_start' and self.kappa_init_param == cpr_step:
                if self.reg_function == 'std':
                    new_kappa = param.std()
                elif self.reg_function == 'l2':
                    new_kappa = param.square().mean()
                kappa.clamp_max_(new_kappa)

            cpr_step += 1
