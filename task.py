import numpy as np
import torch
from torch.utils.data import argument_validation
from tqdm import main
import torchvision


def make_randomcopy_dataset(
    dataset_size, T: int = 500, len_sequence: int = 10, vocab_size: int = 10
):
    assert T >= 2 * len_sequence + 1, "T > 2 * len_sequence"
    flag_pos = np.random.randint(
        low=int(T / 2), high=T - len_sequence - 1, size=[dataset_size], dtype="int64"
    )  # (B)
    x_data = np.eye(T, dtype="int64")[flag_pos] * (vocab_size - 1)  # (B)->One-hot(B, T)
    x_data[:, :len_sequence] = np.random.randint(
        low=1, high=vocab_size - 1, size=[dataset_size, len_sequence]
    )
    y_data = np.zeros([dataset_size, T], dtype="int64")
    flag_pos_start = flag_pos + 1
    indices = np.arange(len_sequence) + flag_pos_start[:, np.newaxis]
    rows = np.arange(dataset_size)[:, None]
    y_data[rows, indices] = x_data[:, :len_sequence]

    x_data_onehot = np.eye(vocab_size)[x_data]
    x_data_onehot_tensor = torch.from_numpy(x_data_onehot).float()
    y_data_tensor = torch.from_numpy(y_data).long()

    return x_data_onehot_tensor, y_data_tensor


class RandomCopyTaskSampler:
    """""" """""" """""" """""
        RandomCopyTask Batch Sampler
        input 112300@0000000 -> output 00000001123000
    """ """""" """""" """""" ""

    def __init__(self, T=500, len_sequence=10, vocab_size=8):
        self.T = T
        self.len_sequence = len_sequence
        self.vocab_size = vocab_size

    def sample_batch(self, batch_size):
        return make_randomcopy_dataset(
            batch_size, self.T, self.len_sequence, self.vocab_size
        )


def make_selectivecopy_dataset(
    dataset_size: int, T: int = 500, len_sequence: int = 10, vocab_size: int = 10
):
    x_data = np.zeros([dataset_size, T], dtype="int64")
    y_data = np.zeros([dataset_size, T], dtype="int64")
    available_indices = np.arange(T - 10, dtype="int64")
    replacement = np.random.randint(
        1, vocab_size - 1, size=[dataset_size, len_sequence], dtype="int64"
    )

    for i in range(dataset_size):
        random_index = np.random.choice(
            available_indices, size=[len_sequence], replace=False
        )
        random_index = np.sort(random_index)  # indexはソートしておく
        x_data[i, random_index] = replacement[i]
        y_data[i, -len_sequence:] = replacement[i]

    x_batch_one_hot = np.eye(vocab_size)[x_data]
    x_batch_one_hot_tensor = torch.from_numpy(x_batch_one_hot).float()
    y_batch_tensor = torch.from_numpy(y_data).long()

    return x_batch_one_hot_tensor, y_batch_tensor


class SelectiveCopyTaskSampler:
    def __init__(self, T, len_sequence, vocab_size):
        self.T = T
        self.len_sequence = len_sequence
        self.vocab_size = vocab_size

    def sample_batch(self, batch_size):
        return make_selectivecopy_dataset(
            batch_size, self.T, self.len_sequence, self.vocab_size
        )


def make_statetransition_dataset(
    dataset_size: int,
    T: int = 4000,
    block_T: int = 400,
    len_sequence: int = 10,
    vocab_size: int = 9,
):
    assert T % block_T == 0, "T must be devided by block_T"
    assert (vocab_size - 1) % 2 == 0, "vocab_size must be odd number"
    assert vocab_size > 5, "vocab_size must be larger than 5"
    NUM_MIN = 3  # 1 is ? copy_a => copy_b, 2 is ! copy_b => copy_a, 0 is common noise
    NUM_MAX = NUM_MIN + (vocab_size - NUM_MIN) // 2 - 1
    ALP_MIN = NUM_MAX + 1
    ALP_MAX = ALP_MIN + (vocab_size - NUM_MIN) // 2 - 1
    x_data = np.zeros([dataset_size, T], dtype="int64")
    y_data = np.zeros([dataset_size, T], dtype="int64")

    # random index
    for i in range(T // block_T):
        x_data[:, i * block_T] = np.random.randint(1, 3, size=[dataset_size])
        available_indices = np.arange(
            i * block_T + 1, i * block_T + block_T, dtype="int64"
        )
        replacement_num = np.random.randint(
            NUM_MIN, NUM_MAX + 1, size=[dataset_size, len_sequence], dtype="int64"
        )
        replacement_alp = np.random.randint(
            ALP_MIN, ALP_MAX + 1, size=[dataset_size, len_sequence], dtype="int64"
        )
        for j in range(dataset_size):
            random_index = np.random.choice(
                available_indices, size=[2, len_sequence], replace=False
            )
            random_index = np.sort(random_index, axis=1)
            x_data[j, random_index[0]] = replacement_num[j]
            x_data[j, random_index[1]] = replacement_alp[j]
            if x_data[j, i * block_T] == 1:
                y_data[j, (i + 1) * block_T - len_sequence : (i + 1) * block_T] = (
                    replacement_num[j] - 2
                )
            elif x_data[j, i * block_T] == 2:
                y_data[j, (i + 1) * block_T - len_sequence : (i + 1) * block_T] = (
                    replacement_alp[j] - 2
                )

    x_batch_one_hot = np.eye(vocab_size)[x_data]
    x_batch_one_hot_tensor = torch.from_numpy(x_batch_one_hot).float()
    y_batch_one_hot = torch.from_numpy(y_data).long()
    return x_batch_one_hot_tensor, y_batch_one_hot


class StateTransitionSampler:
    def __init__(self, T, block_T, len_sequence, vocab_size):
        self.T = T
        self.block_T = block_T
        self.len_sequence = len_sequence
        self.vocab_size = vocab_size

    def sample_batch(self, batch_size):
        return make_statetransition_dataset(
            batch_size, self.T, self.block_T, self.len_sequence, self.vocab_size
        )


def make_copy_dataset(
    dataset_size: int, T: int = 500, len_string: int = 10, vocab_size: int = 10
):
    x_data = np.zeros((dataset_size, T), dtype="int64")
    copy_sequence = np.random.randint(
        low=1, high=vocab_size - 1, size=(dataset_size, len_string)
    )
    x_data[:, :len_string] = copy_sequence
    x_data[:, -(len_string + 1)] = vocab_size - 1
    y_data = np.zeros((dataset_size, T), dtype="int64")
    y_data[:, -len_string:] = x_data[:, :len_string]

    # x_batch_one_hot = np.eye(vocab_size)[x_data]
    x_batch_one_hot = (np.arange(x_data.max() + 1) == x_data[..., None]).astype(int)
    x_batch_one_hot_tensor = torch.from_numpy(x_batch_one_hot).float()
    y_batch_tensor = torch.from_numpy(y_data).long()
    return x_batch_one_hot_tensor, y_batch_tensor


class CopySampler:
    def __init__(self, T, len_sequence, vocab_size):
        self.T = T
        self.len_sequence = len_sequence
        self.vocab_size = vocab_size

    def sample_batch(self, batch_size):
        return make_copy_dataset(batch_size, self.T, self.len_sequence, self.vocab_size)


# def make_cifar10_dataset(batch_size):
#     transform = torchvision.transform(
#         [
#             torchvision.Grayscale(),
#             torchvision.ToTensor(),
#             torchvision.Lambda(lambda x: x.view(-1)),
#         ]
#     )
#     train_sets = torchvision.datasets.CIFAR10(
#         "./", train=True, download=True, transform=transform
#     )
#     eval_sets = torchvision.datasets.CIFAR10(
#         "./", train=False, download=True, transform=transform
#     )
#     train_loader = DataLoader(train_sets, shuffle=True, batch_size=batch_size)
#     eval_loader = DataLoader(eval_sets, shuffle=False, batch_size=batch_size)
#     return


def output_size_of_task(task_name, vocab_size):
    if task_name in ["copy", "randomcopy", "selectivecopy"]:
        return vocab_size - 1
    elif task_name in ["statetransition"]:
        return vocab_size - 2
    else:
        raise ValueError("choose copy, randomcopy, or selectivecopy")


if __name__ == "__main__":
    # test
    # x_batch, y_batch = make_selectivecopy_dataset(10, 20, 5, 8)
    x_batch, y_batch = make_statetransition_dataset(3, 20, 10, 3, 7)
    # x_batch, y_batch = make_copy_dataset(1, 22, 10, 10)
    x_batch = torch.argmax(x_batch, dim=-1)
    print(x_batch[0])
    print(y_batch[0])
    print(x_batch[1])
    print(y_batch[1])
    print(x_batch[2])
    print(y_batch[2])
