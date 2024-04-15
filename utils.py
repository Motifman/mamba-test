import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def make_datasets(input_tensor, target_tensor):
    return TensorDataset(input_tensor, target_tensor)


def make_dataloader(dataset, batch_size: int, shuffle: bool):
    return DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2
    )


def set_seed(seed=123):
    print("set seed numpy and torch")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    np.random.seed(seed)


def accuracy(outputs, targets):
    # outputs: (B, T, D)
    # targets: (B, T)
    outputs = torch.argmax(outputs, dim=2)  # (B, T)
    return (outputs == targets).float().mean()


def accuracy_rc(outputs, targets):
    # outputs: (B, T, D)
    # targets: (B, T)
    outputs = torch.argmax(outputs, dim=2)  # (B, T)
    condition = (targets > 0) & (outputs == targets)
    sum_targets = (targets > 0).float().sum()
    return condition.float().sum() / sum_targets


class Optimizer:
    def __init__(self, parameters, lr, eps=1e-4, opt="adam", use_amp=False, clip=None):
        self._parameters = parameters
        self._opt = {
            "rmsprop": lambda: torch.optim.RMSprop(parameters, lr=lr, eps=eps),
            "adam": lambda: torch.optim.Adam(parameters, lr=lr, eps=eps),
            "nadam": lambda: NotImplemented(f"{opt} is not implemented"),
            "adamax": lambda: torch.optim.Adamax(parameters, lr=lr, eps=eps),
            "sgd": lambda: torch.optim.SGD(parameters, lr=lr),
            "momentum": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9),
        }[opt]()
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self._clip = clip

    def __call__(self, loss):
        assert len(loss.shape) == 0, loss.shape
        self._opt.zero_grad()
        loss.backward()
        if self._clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self._parameters, max_norm=self._clip, norm_type=2
            )
        self._opt.step()


def num_params(model_param):
    sum = 0
    for param in model_param:
        sum += param.data.numel()
    return sum


class EarlyStopping:
    """earlystopping"""

    def __init__(self, patience=3, verbose=False, path="checkpoint_model.pth"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path

    def __call__(self, val_loss, model):
        if np.isnan(val_loss):
            self.early_stop = True
            print("Early stopping due to NaN loss.")
            score = -99999
        else:
            score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.checkpoint(val_loss, model)
            self.counter = 0

    def checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
