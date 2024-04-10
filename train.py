from dataclasses import make_dataclass
from itertools import accumulate
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
import uuid
from tqdm import tqdm
from task import make_randomcopy_dataset, make_selectivecopy_dataset
from utils import (
    make_datasets,
    set_seed,
    make_dataloader,
    accuracy,
    Optimizer,
    num_params,
    EarlyStopping,
)
from model import MambaClassification
import matplotlib.pyplot as plt
import os


def plot_metrics(metrics, path1, path2):
    fig1 = plt.figure()
    plt.plot(metrics["step"], metrics["train_loss"], label="train_loss")
    plt.plot(metrics["step"], metrics["eval_loss"], label="eval_loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.savefig(path1)

    fig2 = plt.figure()
    plt.plot(metrics["step"], metrics["train_acc"], label="train_acc")
    plt.plot(metrics["step"], metrics["eval_acc"], label="eval_acc")
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.savefig(path2)
    return fig1, fig2


def make_datatensor(task_name, n_train, n_eval, T, len_sequence, vocab_size):
    if task_name == "randomcopy":
        x_train_tensor, y_train_tensor = make_randomcopy_dataset(
            n_train, T, len_sequence, vocab_size
        )
        x_eval_tensor, y_eval_tensor = make_randomcopy_dataset(
            n_eval, T, len_sequence, vocab_size
        )
    elif task_name == "selectivecopy":
        x_train_tensor, y_train_tensor = make_selectivecopy_dataset(
            n_train, T, len_sequence, vocab_size
        )
        x_eval_tensor, y_eval_tensor = make_selectivecopy_dataset(
            n_eval, T, len_sequence, vocab_size
        )
    else:
        raise ValueError("You must choose randomcopy or selectivecopy")
    return x_train_tensor, y_train_tensor, x_eval_tensor, y_eval_tensor


def evaluate_best_model(model, train_loader, eval_loader, criterion, device):
    sum_train_loss = 0
    sum_train_acc = 0
    sum_eval_loss = 0
    sum_eval_acc = 0
    len_train = len(train_loader)
    len_eval = len(eval_loader)

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        sum_train_loss += criterion(outputs.transpose(2, 1), labels)
        sum_train_acc += accuracy(outputs, labels)

    for inputs, labels in eval_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        sum_eval_loss += criterion(outputs.transpose(2, 1), labels)
        sum_eval_acc += accuracy(outputs, labels)

    return (
        sum_train_loss / len_train,
        sum_train_acc / len_train,
        sum_eval_loss / len_eval,
        sum_eval_acc / len_eval,
    )


@hydra.main(config_name="config.yaml")
def main(cfg: DictConfig):
    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")
    mlflow.set_experiment(cfg.expname)
    print("=================config.yaml===================")
    print(cfg)

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    run_name = "mamba_" + str(uuid.uuid4())

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("config", config_dict)
        mlflow.log_param("d_model", cfg.model.d_model)
        mlflow.log_param("n_layers", cfg.model.n_layers)
        mlflow.log_param("task_name", cfg.task.name)
        mlflow.log_param("T", cfg.task.T)
        mlflow.log_param("len_sequence", cfg.task.len_sequence)
        mlflow.log_param("vocab_size", cfg.task.vocab_size)
        mlflow.log_param("n_train", cfg.data.n_train)
        mlflow.log_param("n_eval", cfg.data.n_eval)
        mlflow.log_param("batchsize", cfg.data.batch_size)
        mlflow.log_param("grad_steps", cfg.train.grad_steps)
        mlflow.log_param("log_interval", cfg.train.log_interval)
        mlflow.log_param("optim", cfg.optim.name)
        mlflow.log_param("lr", cfg.optim.lr)
        mlflow.log_param("use_amp", cfg.optim.use_amp)
        mlflow.log_param("eps", cfg.optim.eps)
        mlflow.log_param("seed_data", cfg.seed.data)
        mlflow.log_param("seed_train", cfg.seed.train)
        mlflow.log_param("device_id", cfg.device_id)

        device = torch.device(
            f"{cfg.device_id}" if torch.cuda.is_available() else r"cpu"
        )
        print(f"==================device is {device}!!====================")

        # dataset
        set_seed(cfg.seed.data)
        x_train_tensor, y_train_tensor, x_eval_tensor, y_eval_tensor = make_datatensor(
            cfg.task.name,
            cfg.data.n_train,
            cfg.data.n_eval,
            cfg.task.T,
            cfg.task.len_sequence,
            cfg.task.vocab_size - 2,
        )
        train_sets = make_datasets(x_train_tensor, y_train_tensor)
        eval_sets = make_datasets(x_eval_tensor, y_eval_tensor)
        train_loader = make_dataloader(train_sets, cfg.data.batch_size, shuffle=True)
        eval_loader = make_dataloader(eval_sets, cfg.data.batch_size, shuffle=False)
        train_loader_iterator = iter(train_loader)
        eval_loader_iterator = iter(eval_loader)

        # model, optimiezer, criterion, EarlyStopping
        set_seed(cfg.seed.train)

        # assert (
        #     cfg.model.input_size == cfg.task.vocab_size
        # ), "plz d_model == vocab_size"
        model = MambaClassification(
            cfg.model.input_size,
            cfg.model.d_model,
            cfg.model.n_layers,
            cfg.task.vocab_size,
        ).to(device)
        n_param = num_params(model.parameters())
        print(f"paramters = {n_param}")
        mlflow.log_param("n_parameter", n_param)
        optimiezer = Optimizer(
            model.parameters(),
            cfg.optim.lr,
            eps=cfg.optim.eps,
            opt=cfg.optim.name,
            use_amp=cfg.optim.use_amp,
        )
        criterion = nn.CrossEntropyLoss()
        script_dir = os.path.dirname(os.path.realpath(__file__))
        earlystopping = EarlyStopping(
            patience=10, verbose=False, path=script_dir + "/log/checkpoint_model.pth"
        )

        # metrics
        metrics = {
            "step": [],
            "train_loss": [],
            "train_acc": [],
            "eval_loss": [],
            "eval_acc": [],
        }

        # train loop
        sum_train_loss = 0
        sum_train_acc = 0
        sum_eval_loss = 0
        sum_eval_acc = 0
        GRAD_STEPS = cfg.train.grad_steps
        LOG_INTERVAL = cfg.train.log_interval
        itr = 0
        for step in tqdm(range(GRAD_STEPS)):
            # train
            model.train()
            try:
                inputs, labels = next(train_loader_iterator)
            except StopIteration:
                train_loader_iterator = iter(train_loader)
                inputs, labels = next(train_loader_iterator)

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.transpose(1, 2), labels)  # (B, T, C)->(B, C, T)
            optimiezer(loss)
            train_acc = accuracy(outputs, labels)
            sum_train_loss += loss.item()
            sum_train_acc += train_acc.item()

            # eval
            model.eval()
            with torch.no_grad():
                try:
                    inputs, labels = next(eval_loader_iterator)
                except StopIteration:
                    eval_loader_iterator = iter(eval_loader)
                    inputs, labels = next(eval_loader_iterator)

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.transpose(1, 2), labels)
                eval_acc = accuracy(outputs, labels)
                sum_eval_loss += loss.item()
                sum_eval_acc += eval_acc.item()

            itr += 1

            if step % LOG_INTERVAL == 0:
                # metrics
                train_loss = sum_train_loss / itr
                train_acc = sum_train_acc / itr
                eval_loss = sum_eval_loss / itr
                eval_acc = sum_eval_acc / itr

                print(f"train_loss={train_loss}, train_acc={train_acc}")
                print(f"eval_loss={eval_loss}, eval_acc={eval_acc}")
                sum_train_loss = 0
                sum_train_acc = 0
                sum_eval_loss = 0
                sum_eval_acc = 0

                metrics["step"].append(step)
                metrics["train_loss"].append(train_loss)
                metrics["train_acc"].append(train_acc)
                metrics["eval_loss"].append(eval_loss)
                metrics["eval_acc"].append(eval_acc)

                mlflow.log_metric("train_loss", train_loss, step=step)
                mlflow.log_metric("train_acc", train_acc, step=step)
                mlflow.log_metric("eval_loss", eval_loss, step=step)
                mlflow.log_metric("eval_acc", eval_acc, step=step)

                earlystopping(eval_loss, model=model)
                if (
                    earlystopping.early_stop
                ):  # ストップフラグがTrueの場合、breakでforループを抜ける
                    print(
                        f"Early Stopping! best eval loss is {earlystopping.best_score}"
                    )
                    mlflow.log_metric("stop epoch", step)
                    break
                itr = 0

        # log
        log_dir = "log/"
        os.makedirs(log_dir, exist_ok=True)
        path1 = log_dir + "loss.pdf"
        path2 = log_dir + "acc.pdf"
        fig1, fig2 = plot_metrics(metrics, path1, path2)
        mlflow.log_figure(fig1, path1)
        mlflow.log_figure(fig2, path2)

        # evaluate the best model
        train_loss, train_acc, eval_loss, eval_acc = evaluate_best_model(
            model, train_loader, eval_loader, criterion, device
        )
        print("evaluate best model")
        print(f"train_loss={train_loss}, train_acc={train_acc}")
        print(f"eval_loss={eval_loss}, eval_acc={eval_acc}")
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("train_acc", train_acc)
        mlflow.log_metric("eval_loss", eval_loss)
        mlflow.log_metric("eval_acc", eval_acc)


if __name__ == "__main__":
    main()
