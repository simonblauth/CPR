import os
from argparse import ArgumentParser
from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from pytorch_cpr import apply_CPR

### Data
def modular_addition(p, train_fraction, train_shuffle, device):
    equals_token = p
    x, y = torch.meshgrid(torch.arange(p), torch.arange(p), indexing='ij')
    x = x.flatten()
    y = y.flatten()
    equals = torch.ones(x.shape, dtype=torch.int64) * equals_token
    prompts = torch.stack([x, y, equals], dim=1).to(device)
    answers = ((x + y) % p).to(device)

    data = torch.utils.data.TensorDataset(prompts, answers)
    train, test = torch.utils.data.random_split(data,
                                                [int(train_fraction * len(data)),
                                                 len(data) - int(train_fraction * len(data))
                                                 ])

    train_loader = torch.utils.data.DataLoader(train, batch_size=512, shuffle=train_shuffle)
    test_loader = torch.utils.data.DataLoader(test, batch_size=len(data), shuffle=False)
    return train_loader, test_loader


### Model
class Block(nn.Module):
    def __init__(self, dim, num_heads, use_ln):
        super().__init__()
        self.use_ln = use_ln
        if use_ln:
            self.ln_1 = nn.LayerNorm(dim)
            self.ln_2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, bias=False)
        activation = nn.ReLU()
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), activation, nn.Linear(dim * 4, dim), )

    def forward(self, x):
        attn_mask = torch.full((len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        if self.use_ln:
            x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        if self.use_ln:
            x = x + self.mlp(self.ln_2(x))
        else:
            x = x + self.mlp(x)
        return x

class TransformerDecoder(nn.Module):

    def __init__(self, dim, num_layers, num_tokens, num_heads=4, seq_len=3, use_ln=False):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.position_embeddings = nn.Embedding(seq_len, dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(dim, num_heads, use_ln))
        self.use_ln = use_ln
        if use_ln:
            self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)

    def forward(self, x):
        h = self.token_embeddings(x)
        positions = torch.arange(x.shape[0], device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)
        for layer in self.layers:
            h = layer(h)
        if self.use_ln:
            h = self.ln_f(h)
        logits = self.head(h)
        return logits


def init_params(model, model_dim, vocab_dim, init_type='xavier'):
    for name, param in model.named_parameters():
        if param.dim() > 1:
            if vocab_dim in param.shape:
                nn.init.normal_(param, 0, 1 / np.sqrt(vocab_dim))
            else:
                if init_type == 'xavier':
                    nn.init.xavier_normal_(param)
                elif init_type == 'sqrt_dim':
                    nn.init.normal_(param, 0, 1 / np.sqrt(model_dim))
        else:
            nn.init.constant_(param, 0)


def print_param_groups(param_groups):
    for param_group in param_groups:
        if 'apply_decay' in param_group:
            print(f"### PARAM GROUP #### apply_decay: {param_group['apply_decay']}")
        else:
            print(f"### PARAM GROUP #### weight_decay: {param_group['weight_decay']}")
        for name, param in zip(param_group['names'], param_group['params']):
            print(
                f"{name:60} {param.shape[0]:4} {param.shape[-1]:4} std {param.std():.3f} l2m {param.square().mean():.3f}")


### Main
def train_grokking(config):
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    print("Config", config)

    if config.device is not None:
        device = config.device
        print("starting on device", device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = modular_addition(config.p, train_fraction=config.train_fraction,
                                                 train_shuffle=config.train_shuffle, device=device)

    model = TransformerDecoder(
        dim=config.model_dim, num_layers=config.num_layers, num_heads=config.num_heads, num_tokens=config.p + 1,
        seq_len=3, use_ln=config.use_ln).to(device)

    init_params(model, config.model_dim, config.p, init_type=config.init_type)

    if config.optimizer == 'adamcpr':
        optimizer = apply_CPR(model, torch.optim.Adam, config.kappa_init_param, config.kappa_init_method,
                              config.reg_function,
                              config.kappa_adapt, config.kappa_update,
                              normalization_regularization=False, bias_regularization=False,
                              embedding_regularization=True,
                              lr=config.lr, betas=(config.beta1, config.beta2))
        param_groups = optimizer.state_dict()['param_groups']
        params = list(model.parameters())
        for param_group in param_groups:
            for index, param_id in enumerate(param_group['params']):
                param_group['params'][index] = params[param_id]
    else:
        param_dict = {pn: p for pn, p in model.named_parameters()
                      if p.requires_grad}
        if config.exclude_reg is not None:
            exclude_reg = config.exclude_reg.split(",")
            param_groups = [{"params": [], "names": [], 'weight_decay': config.weight_decay}, {
                "params": [], "names": [], 'weight_decay': 0}]
            for k, v in param_dict.items():
                print(k)
                if any([reg in k for reg in exclude_reg]):
                    param_groups[1]["params"].append(v)
                    param_groups[1]["names"].append(k)
                else:
                    param_groups[0]["params"].append(v)
                    param_groups[0]["names"].append(k)

        else:
            param_groups = model.parameters()
        optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(config.beta1, config.beta2))

    if config.print:
        print_param_groups(param_groups)

    if config.rescale_alpha > 0:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n.endswith("weight"):
                    p.data *= config.rescale_alpha
            norm = np.sqrt(sum(p.pow(2).sum().item() for n, p in model.named_parameters() if n.endswith("weight")))

    stats = defaultdict(list)
    steps = 0

    test_x, test_labels = next(iter(test_loader)) # ther is only one tests batch
    test_x, test_labels = test_x.H.to(device), test_labels.to(device)

    for epoch in tqdm(range(config.epochs), disable=not config.print):

        for train_x, train_labels in train_loader:

            model.train(True)
            train_x, train_labels = train_x.H.to(device), train_labels.to(device)

            train_logits = model(train_x)
            train_loss = torch.nn.functional.cross_entropy(train_logits[-1], train_labels)

            model.zero_grad()
            train_loss.backward()
            optimizer.step()

            if config.rescale_alpha > 0:
                with torch.no_grad():
                    new_norm = np.sqrt(
                        sum(p.pow(2).sum().item() for n, p in model.named_parameters() if n.endswith("weight")))
                    for n, p in model.named_parameters():
                        if n.endswith("weight"):
                            p.data *= norm / new_norm

        if epoch % config.log_interval == 0:
            with torch.no_grad():

                model.train(False)
                test_logits = model(test_x).detach()
                test_loss = torch.nn.functional.cross_entropy(test_logits[-1], test_labels)

                train_acc = (train_logits[-1].argmax(-1) == train_labels).float().mean()
                test_acc = (test_logits[-1].argmax(-1) == test_labels).float().mean()

                stats['train_loss'].append(train_loss.cpu().numpy())
                stats['val_loss'].append(test_loss.cpu().numpy())
                stats['train_acc'].append(train_acc.cpu().numpy())
                stats['val_acc'].append(test_acc.cpu().numpy())
                stats['total_norm'].append(
                    torch.sqrt(sum(param.pow(2).sum() for param in model.parameters())).cpu().numpy())
                stats['steps'].append(steps)

                if config.optimizer == "adamcpr":
                    for group, group_states in zip(optimizer.base_optim.param_groups, optimizer.cpr_states):
                        if 'apply_decay' in group and group['apply_decay'] is True:
                            for name, state in zip(group['names'], group_states):
                                lagmul = state['lagmul']
                                kappa = state['kappa']
                                step = state['step']
                                stats[f"cpr/{name}/lambda"].append(lagmul.item())
                                stats[f"cpr/{name}/kappa"].append(kappa.item())
                                stats[f"cpr/{name}/step"].append(step.item())

                totalnorm = []
                for param_group in optimizer.param_groups:
                    for name, param in zip(param_group['names'], param_group['params']):
                        stats[f"params/{name}/mean"].append(param.mean().item())
                        stats[f"params/{name}/std"].append(param.std().item())
                        stats[f"params/{name}/l2"].append(param.pow(2).sum().item())
                        stats[f"params/{name}/l2m"].append(param.pow(2).mean().item())
                        stats[f"params/{name}/l2s"].append(param.pow(2).sum().item())
                        totalnorm.append(param.pow(2).sum().item())
                stats[f"params/total_norm"].append(np.sqrt(sum(totalnorm)))

        steps += 1

    task_name = f"{config.epochs}_{str(int(config.seed))}_p{config.p}_f{config.train_fraction}"
    if config.optimizer == "adamcpr":
        expt_name = f"{config.optimizer}_p{config.kappa_init_param}_m{config.kappa_init_method}_kf{config.reg_function}_r{config.kappa_update}_l{config.lr}_adapt{config.kappa_adapt}"
    else:
        expt_name = f"{config.optimizer}_w{config.weight_decay}_re{config.rescale_alpha}_l{config.lr}"

    config.output_dir = f"{config.output_dir}/grokking_{task_name}"
    os.makedirs(config.output_dir, exist_ok=True)
    config_dict = config.__dict__
    if config.print:
        print(expt_name, config_dict)

    os.makedirs(config.output_dir + f"/{config.session}_stats", exist_ok=True)
    np.save(f"{config.output_dir}/{config.session}_stats/{expt_name}.npy",
            {"name": expt_name, 'stats': stats, 'config': config_dict})

    if config.plot:
        os.makedirs(config.output_dir + f"/{config.session}_figures", exist_ok=True)

        if config.plot_norms:
            name_constrained_weights = param_groups[0]['names']
            plot_rows = 1 + len(name_constrained_weights)

            fig, ax = plt.subplots(plot_rows, 1, sharex=True, squeeze=True, figsize=(16, 12))

            ax[0].plot(stats['steps'], stats['train_acc'], color='red', label="train")
            ax[0].plot(stats['steps'], stats['val_acc'], color='green', label="val")
            ax[0].legend()
            ax[0].set_ylabel("Accuracy")
            ax[0].set_xlim(8, 2 * config.epochs)
            ax[0].set_xscale("log", base=10)
            ax[0].set_title(expt_name)

            for idx, name in enumerate(name_constrained_weights):
                axr = idx + 1
                ax[axr].plot(stats['steps'], stats[f"params/{name}/std"], color='orange', label=f"std {name}")
                ax[axr].set_ylabel("STD")
                ax2 = ax[axr].twinx()
                if f"cpr/{name}/lambda" in stats.keys():
                    ax2.plot(stats['steps'], stats[f"cpr/{name}/lambda"], color='purple', label=f"lambda {name}")
                    ax2.set_ylabel("Lambda", color='purple')
                else:
                    ax2.plot(stats['steps'], stats[f"params/{name}/l2m"], color='purple', label=f"l2m {name}")
                    ax2.set_ylabel("Weight Norm", color='purple')
                ax[axr].set_xlim(8, 2 * config.epochs)
                ax[axr].set_xscale("log", base=10)
                ax[axr].legend(loc=(0.015, 0.72))
                ax[axr].legend()
                if idx < len(name_constrained_weights) - 1:
                    plt.setp(ax[axr].get_xticklabels(), visible=False)
            ax[axr].set_xlabel("Optimization Steps")
            fig.subplots_adjust(0.08, 0.1, 0.95, 0.93, 0, 0)

        else:
            ax = plt.subplot(1, 1, 1)
            plt.plot(stats['steps'], stats['train_acc'], color='red', label="train")
            plt.plot(stats['steps'], stats['val_acc'], color='green', label="val")
            plt.legend()
            plt.xlabel("Optimization Steps")
            plt.ylabel("Accuracy")
            plt.xlim(8, 2 * config.epochs)
            ax2 = ax.twinx()
            if f"cpr/{name}/lambda" in stats.keys():
                ax2.plot(stats['steps'], stats[f"cpr/{name}/lambda"], color='purple', label=f"lambda {name}")
                ax2.set_ylabel("Lambda", color='purple')
            else:
                ax2.plot(stats['steps'], stats[f"params/{name}/l2m"], color='purple', label=f"l2m {name}")
                ax2.set_ylabel("Weight Norm", color='purple')
            ax2.set_ylim(27, 63)
            plt.xscale("log", base=10)
            plt.legend(loc=(0.015, 0.72))
            plt.tight_layout()
            plt.title(expt_name)

        plt.savefig(f"{config.output_dir}/{config.session}_figures/{expt_name}.png", dpi=150)

        if config.show_plot:
            plt.show()

        plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--session", type=str, default='test_grokking')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=4000)

    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--train_shuffle", type=bool, default=True)
    parser.add_argument("--train_fraction", type=float, default=0.3)

    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--use_ln", type=bool, default=False)
    parser.add_argument("--init_type", type=str, default='sqrt_dim')

    parser.add_argument("--optimizer", type=str, default='adamcpr')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--exclude_reg", type=str, default='bias,norm')

    parser.add_argument("--rescale_alpha", type=float, default=0)

    parser.add_argument("--kappa_init_param", type=float, default=0.8)
    parser.add_argument("--kappa_init_method", type=str, default='dependent')
    parser.add_argument("--reg_function", type=str, default='l2')
    parser.add_argument("--kappa_update", type=float, default=1.0)
    parser.add_argument("--kappa_adapt", type=bool, default=True)

    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default='grokking')
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--show_plot", type=bool, default=True)
    parser.add_argument("--print", type=bool, default=True)
    parser.add_argument("--plot_norms", type=bool, default=True)
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    print(args.__dict__)

    if args.rescale_alpha > 0:
        assert args.optimizer == 'adamw'

    train_grokking(args)
