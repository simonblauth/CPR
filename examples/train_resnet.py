import pathlib, argparse
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import LearningRateMonitor

from pytorch_cpr import apply_CPR

torch.set_float32_matmul_precision('high')

class WeightDecayScheduler(Callback):

    def __init__(self, schedule_weight_decay: bool, schedule_type: str, scale: float):
        super().__init__()
        self.schedule_weight_decay = schedule_weight_decay

        self.schedule_type = schedule_type

        self.decay = scale

        self._step_count = 0

    @staticmethod
    def get_scheduler(schedule_type, num_warmup_steps, decay_factor, num_training_steps):
        def fn_scheduler(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            elif schedule_type == 'linear':
                return (decay_factor + (1 - decay_factor) *
                        max(0.0, float(num_training_steps - num_warmup_steps - current_step) / float(
                            max(1, num_training_steps - num_warmup_steps))))
            elif schedule_type == 'cosine':
                return (decay_factor + (1 - decay_factor) *
                        max(0.0, (1 + math.cos(math.pi * (current_step - num_warmup_steps) / float(
                            max(1, num_training_steps - num_warmup_steps)))) / 2))
            elif schedule_type == 'const':
                return 1.0

        return fn_scheduler

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self.num_training_steps = trainer.max_steps

        self.weight_decay = []
        for optim in trainer.optimizers:
            for group_idx, group in enumerate(optim.param_groups):
                if 'weight_decay' in group:
                    self.weight_decay.append(group['weight_decay'])

        num_warmup_steps = 0

        self.scheduler = self.get_scheduler(self.schedule_type, num_warmup_steps, self.decay, self.num_training_steps)

    def on_before_optimizer_step(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer):

        if self.schedule_weight_decay:
            stats = {}
            for group_idx, group in enumerate(optimizer.param_groups):
                if 'weight_decay' in group:
                    group['weight_decay'] = self.weight_decay[group_idx] * self.scheduler(self._step_count)
                    stats[f"weight_decay/rank_{trainer.local_rank}/group_{group_idx}"] = group['weight_decay']

            if trainer.loggers is not None:
                for logger in trainer.loggers:
                    logger.log_metrics(stats, step=trainer.global_step)
            self._step_count += 1


### Data
def cifar100_task(cache_dir='./data'):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(root=cache_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=cache_dir, train=False, download=True, transform=transform_test)

    return trainset, testset


def wd_group_named_parameters(model, weight_decay):
    apply_decay = set()
    apply_no_decay = set()
    special = set()
    whitelist_weight_modules = (nn.Linear, nn.Conv2d, nn.Embedding)
    blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                nn.LazyBatchNorm1d, nn.LazyBatchNorm2d, nn.LazyBatchNorm3d,
                                nn.GroupNorm, nn.SyncBatchNorm,
                                nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                                nn.LayerNorm, nn.LocalResponseNorm)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            if not p.requires_grad or fpn not in param_dict:
                continue  # frozen weights
            if hasattr(p, '_optim'):
                special.add(fpn)
            elif pn.endswith('bias'):
                apply_no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                apply_decay.add(fpn)
            elif isinstance(m, blacklist_weight_modules):
                apply_no_decay.add(fpn)
            else:
                print("cpr_group_named_parameters: Not using any rule for ", fpn, " in ", type(m))

    apply_decay |= (param_dict.keys() - apply_no_decay - special)

    # validate that we considered every parameter
    inter_params = apply_decay & apply_no_decay
    union_params = apply_decay | apply_no_decay
    assert len(inter_params) == 0, f"Parameters {str(inter_params)} made it into both apply_decay/apply_no_decay sets!"
    assert len(
        param_dict.keys() - special - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)}  were not separated into either apply_decay/apply_no_decay set!"

    if not apply_no_decay:
        param_groups = [{"params": [param_dict[pn] for pn in sorted(list(apply_no_decay | apply_decay))],
                         "names": [pn for pn in sorted(list(apply_no_decay | apply_decay))],
                         "weight_decay": weight_decay}]
    else:
        param_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(apply_decay))],
             "names": [pn for pn in sorted(list(apply_decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(apply_no_decay))],
             "names": [pn for pn in sorted(list(apply_no_decay))], "weight_decay": 0.0},
        ]

    return param_groups


### Model
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


### Lightning Module
class ResNetModule(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.cfg = config

        if self.cfg.model_name == "ResNet18":
            self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100)
        elif self.cfg.model_name == "ResNet34":
            self.model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=100)
        elif self.cfg.model_name == "ResNet50":
            self.model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=100)
        elif self.cfg.model_name == "ResNet101":
            self.model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=100)
        elif self.cfg.model_name == "ResNet152":
            self.model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=100)

        self.loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.test_stats = []

    def configure_optimizers(self):

        if self.cfg.optimizer == 'adamw':
            param_groups = wd_group_named_parameters(self.model, weight_decay=self.cfg.weight_decay)
            optimizer = torch.optim.AdamW(param_groups, lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2))
        elif self.cfg.optimizer == 'adamcpr':
            optimizer = apply_CPR(self.model, torch.optim.Adam, self.cfg.kappa_init_param, self.cfg.kappa_init_method,
                                  self.cfg.reg_function,
                                  self.cfg.kappa_adapt, self.cfg.kappa_update,
                                  embedding_regularization=True,
                                  lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2))

        if self.cfg.rescale_alpha > 0.0:
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if n.endswith("weight"):
                        p.data *= self.cfg.rescale_alpha
                self.rescale_norm = np.sqrt(
                    sum(p.pow(2).sum().item() for n, p in self.model.named_parameters() if n.endswith("weight")))

        lr_decay_factor = self.cfg.lr_decay_factor
        num_warmup_steps = self.cfg.lr_warmup_steps

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                return lr_decay_factor + (1 - lr_decay_factor) * max(0.0, (1 + math.cos(
                    math.pi * (current_step - num_warmup_steps) / float(
                        max(1, self.cfg.max_train_steps - num_warmup_steps)))) / 2)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

        return [optimizer], {'scheduler': lr_scheduler, 'interval': 'step'}

    def setup(self, stage: str) -> None:
        trainset, testset = cifar100_task()
        self.trainset = trainset
        self.testset = testset

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=self.cfg.batch_size, shuffle=True,
                                                   num_workers=8)
        return train_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(self.testset, batch_size=self.cfg.batch_size, shuffle=False,
                                                  num_workers=8)
        return test_loader

    def _accuracy(self, y_hat, y):
        return torch.sum(torch.argmax(y_hat, dim=1) == y).item() / len(y)

    def training_step(self, batch, batch_idx):

        X, y = batch
        y_hat = self.model(X)
        loss = self.loss(y_hat, y)

        self.log('train_loss', loss)

        if self.cfg.rescale_alpha > 0.0:
            with torch.no_grad():
                new_norm = np.sqrt(
                    sum(p.pow(2).sum().item() for n, p in self.model.named_parameters() if n.endswith("weight")))
                for n, p in self.model.named_parameters():
                    if n.endswith("weight"):
                        p.data *= self.rescale_norm / new_norm
        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model(X)
        loss = self.loss(y_hat, y)

        correct_pred = torch.sum(torch.argmax(y_hat, dim=1) == y).item()
        num_samples = len(y)
        self.test_stats.append({'loss': loss.item(), 'correct_pred': correct_pred, 'num_samples': num_samples})
        self.log('test_loss', loss)

        return loss

    def on_test_epoch_end(self):
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])

        valid_loss = np.mean([s['loss'] for s in self.test_stats])
        valid_accuracy = np.sum([s['correct_pred'] for s in self.test_stats]) / np.sum(
            [s['num_samples'] for s in self.test_stats])
        self.log('test_loss', valid_loss)
        self.log('test_accuracy', valid_accuracy)
        self.test_stats = []


def train_cifar100_task(config):
    task_name = f"{config.model_name}_seed{config.seed}_steps{config.max_train_steps}"
    expt_dir = pathlib.Path(config.output_dir) / config.session / task_name
    expt_dir.mkdir(parents=True, exist_ok=True)

    if config.optimizer == "adamcpr":
        expt_name = f"{config.optimizer}_p{config.kappa_init_param}_m{config.kappa_init_method}_kf{config.reg_function}_r{config.kappa_update}_l{config.lr}_adapt{config.kappa_adapt}"
    else:
        expt_name = f"{config.optimizer}_l{config.lr}_w{config.weight_decay}_re{config.rescale_alpha}_swd{config.schedule_weight_decay}_swds{config.wd_scale}_t{config.wd_schedule_type}"

    (expt_dir / expt_name).mkdir(parents=True, exist_ok=True)
    np.save(expt_dir / expt_name / "config.npy", config.__dict__)
    logger = TensorBoardLogger(save_dir=expt_dir, name=expt_name)
    pl.seed_everything(config.seed)

    if config.device:
        devices = [config.device]
    else:
        devices = [0]

    model = ResNetModule(config)

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        WeightDecayScheduler(config.schedule_weight_decay, schedule_type=config.wd_schedule_type, scale=config.wd_scale)
    ]

    trainer = pl.Trainer(devices=devices, accelerator="gpu", max_steps=config.max_train_steps,
                         log_every_n_steps=config.log_interval,
                         enable_progress_bar=config.enable_progress_bar,
                         logger=logger, callbacks=callbacks)
    trainer.fit(model)

    # evaluate model
    result = trainer.test(model)
    np.save(expt_dir / expt_name / "result.npy", result)
    print(result)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--session", type=str, default='test_resnet')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_name", type=str, default="ResNet18")
    parser.add_argument("--optimizer", type=str, default='adamcpr')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.1)

    parser.add_argument("--schedule_weight_decay", action=argparse.BooleanOptionalAction)
    parser.add_argument("--wd_schedule_type", type=str, default='cosine')
    parser.add_argument("--wd_scale", type=float, default=0.1)

    parser.add_argument("--lr_warmup_steps", type=int, default=200)
    parser.add_argument("--lr_decay_factor", type=float, default=0.1)
    parser.add_argument("--rescale_alpha", type=float, default=0)

    parser.add_argument("--kappa_init_param", type=float, default=1000)
    parser.add_argument("--kappa_init_method", type=str, default='warm_start')
    parser.add_argument("--reg_function", type=str, default='l2')
    parser.add_argument("--kappa_update", type=float, default=1.0)
    parser.add_argument("--kappa_adapt", action=argparse.BooleanOptionalAction)

    parser.add_argument("--start_epoch", type=int, default=1)

    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--enable_progress_bar", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default='cifar100')
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    args.schedule_weight_decay = args.schedule_weight_decay == 1
    args.kappa_adapt = args.kappa_adapt == 1

    print(args.__dict__)

    if args.rescale_alpha > 0.0:
        assert args.optimizer == 'adamw'

    train_cifar100_task(args)
