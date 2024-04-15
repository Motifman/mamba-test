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
from task import (
    make_randomcopy_dataset,
    make_selectivecopy_dataset,
    make_statetransition_dataset,
    make_copy_dataset,
)
from utils import (
    make_datasets,
    set_seed,
    make_dataloader,
    accuracy,
    accuracy_rc,
    Optimizer,
    num_params,
    EarlyStopping,
)
from model import make_model
import matplotlib.pyplot as plt
import os


def make_datatensor(task_name, n_train, n_eval, T, block_T, len_sequence, vocab_size):
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
    elif task_name == "statetransition":
        x_train_tensor, y_train_tensor = make_statetransition_dataset(
            n_train, T, block_T, len_sequence, vocab_size
        )
        x_eval_tensor, y_eval_tensor = make_statetransition_dataset(
            n_eval, T, block_T, len_sequence, vocab_size
        )
    elif task_name == "copy":
        x_train_tensor, y_train_tensor = make_copy_dataset(
            n_train, T, len_sequence, vocab_size
        )
        x_eval_tensor, y_eval_tensor = make_copy_dataset(
            n_eval, T, len_sequence, vocab_size
        )
    else:
        raise ValueError(
            "You must choose randomcopy or selectivecopy or statetransition"
        )
    return x_train_tensor, y_train_tensor, x_eval_tensor, y_eval_tensor


def copy_accuracy(outputs, labels):
    pred = torch.max(outputs, dim=2)[1]  # argmax
    acc_seq = pred[:, -10:] == labels[:, -10:]
    return torch.mean(torch.sum(acc_seq, dim=1).cpu() / 10)


@hydra.main(config_name="config.yaml")
def main(cfg: DictConfig):
    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")
    mlflow.set_experiment(cfg.expname)
    print("=================config.yaml===================")
    print(cfg)

    # config_dict = OmegaConf.to_container(cfg, resolve=True)
    run_name = "mamba_" + str(uuid.uuid4())

    with mlflow.start_run(run_name=run_name):
        # mlflow.log_param("config", config_dict)
        mlflow.log_param("model_name", cfg.model.name)
        mlflow.log_param("d_model", cfg.model.d_model)
        mlflow.log_param("n_layers", cfg.model.n_layers)
        mlflow.log_param("parallel", cfg.model.parallel)
        mlflow.log_param("activation", cfg.model.activation)
        mlflow.log_param("task_name", cfg.task.name)
        mlflow.log_param("T", cfg.task.T)
        mlflow.log_param("len_sequence", cfg.task.len_sequence)
        mlflow.log_param("vocab_size", cfg.task.vocab_size)
        mlflow.log_param("n_train", cfg.data.n_train)
        mlflow.log_param("n_eval", cfg.data.n_eval)
        mlflow.log_param("batchsize", cfg.data.batch_size)
        mlflow.log_param("grad_steps", cfg.train.grad_steps)
        mlflow.log_param("patience", cfg.train.patience)
        mlflow.log_param("log_interval", cfg.train.log_interval)
        mlflow.log_param("clip", cfg.optim.clip)
        mlflow.log_param("optim", cfg.optim.name)
        mlflow.log_param("lr", cfg.optim.lr)
        mlflow.log_param("use_amp", cfg.optim.use_amp)
        mlflow.log_param("eps", cfg.optim.eps)
        mlflow.log_param("seed_data", cfg.seed.data)
        mlflow.log_param("seed_train", cfg.seed.train)
        mlflow.log_param("device_id", cfg.device_id)
        mlflow.log_param("eval_only", cfg.eval_only)
        mlflow.log_param("param_count_only", cfg.param_count_only)

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
            cfg.task.block_T,
            cfg.task.len_sequence,
            cfg.task.vocab_size,
        )
        train_sets = make_datasets(x_train_tensor, y_train_tensor)
        eval_sets = make_datasets(x_eval_tensor, y_eval_tensor)
        train_loader = make_dataloader(train_sets, cfg.data.batch_size, shuffle=True)
        eval_loader = make_dataloader(eval_sets, cfg.data.batch_size, shuffle=False)

        # model, optimiezer, criterion, EarlyStopping
        set_seed(cfg.seed.train)
        model = make_model(
            cfg.model.name,
            cfg.model.input_size,
            cfg.model.d_model,
            cfg.model.n_layers,
            cfg.task.vocab_size - 1,
            cfg.model.parallel,
            cfg.model.activation,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # if cfg.checkpoint_model is not None:
        #     print("loading pretrained model...")
        #     path = script_dir + f"/{cfg.log_dir}/{cfg.checkpoint_model}"
        #     model.load_state_dict(torch.load(path))
        #
        if cfg.param_count_only:
            n_param = num_params(model.parameters())
            print(f"paramters = {n_param}")
            mlflow.log_param("n_parameter", n_param)
            quit()

        criterion = nn.CrossEntropyLoss()
        earlystopping = EarlyStopping(
            patience=cfg.train.patience,
            verbose=False,
            path=script_dir + f"/log/{run_name}_checkpoint_model.pth",
        )

        # metrics
        metrics = {
            "step": [],
            "train_loss": [],
            "train_acc": [],
            "eval_loss": [],
            "eval_acc": [],
        }

        # eval_only
        if cfg.eval_only:
            with torch.no_grad():
                train_loss, train_acc, eval_loss, eval_acc = evaluate_best_model(
                    model, train_loader, eval_loader, criterion, device, accuracy_rc
                )
            print("evaluate best model")
            print(f"best_train_loss={train_loss}, best_train_acc={train_acc}")
            print(f"best_eval_loss={eval_loss}, best_eval_acc={eval_acc}")
            mlflow.log_metric("best_train_loss", train_loss)
            mlflow.log_metric("best_train_acc", train_acc)
            mlflow.log_metric("best_eval_loss", eval_loss)
            mlflow.log_metric("best_eval_acc", eval_acc)
            quit()

        # train loop
        EPOCH = 150
        for epoch in tqdm(range(EPOCH)):
            sum_train_loss = 0
            sum_train_acc = 0
            sum_train_acc_test = 0
            # train
            model.train()
            for inputs, labels in train_loader:
                model.init_states()
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(
                    outputs.transpose(1, 2), labels
                )  # (B, T, C)->(B, C, T)
                # optimizer(loss)
                train_acc = accuracy_rc(outputs, labels)
                sum_train_loss += loss.item()
                sum_train_acc += train_acc.item()
                sum_train_acc_test += copy_accuracy(outputs, labels).item()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=10, norm_type=2
                )
                optimizer.step()

            train_loss = sum_train_loss / len(train_loader)
            train_acc = sum_train_acc / len(train_loader)
            train_acc_test = sum_train_acc_test / len(train_loader)

            print(f"Epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}")
            print(f"accuracy_test={train_acc_test}")

            metrics["step"].append(epoch)
            metrics["train_loss"].append(train_loss)
            metrics["train_acc"].append(train_acc)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)

            earlystopping(train_loss, model=model)
            if earlystopping.early_stop:
                print(f"Early Stopping! best eval loss is {earlystopping.best_score}")
                mlflow.log_metric("stop epoch", epoch)
                break

        # log
        # path1 = script_dir + f"/{cfg.log_dir}/loss.pdf"
        # path2 = script_dir + f"/{cfg.log_dir}/acc.pdf"
        # fig1, fig2 = plot_metrics(metrics, path1, path2)
        # mlflow.log_figure(fig1, "loss.pdf")
        # mlflow.log_figure(fig2, "acc.pdf")

        # evaluate the best model
        # with torch.no_grad():
        #     train_loss, train_acc, eval_loss, eval_acc = evaluate_best_model(
        #         model, train_loader, eval_loader, criterion, device, accuracy_rc
        #     )
        # print("evaluate best model")
        # print(f"best_train_loss={train_loss}, best_train_acc={train_acc}")
        # print(f"best_eval_loss={eval_loss}, best_eval_acc={eval_acc}")
        # mlflow.log_metric("best_train_loss", train_loss)
        # mlflow.log_metric("best_train_acc", train_acc)
        # mlflow.log_metric("best_eval_loss", eval_loss)
        # mlflow.log_metric("best_eval_acc", eval_acc)


if __name__ == "__main__":
    main()
