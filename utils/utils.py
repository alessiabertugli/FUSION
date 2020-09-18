import numpy as np
import torch
import random


def find_index(y_traj, y_rand):
    for idx, label in enumerate(y_rand[0]):
        if label == y_traj[0]:
            return idx

def sample_balanced_data(cactus_partition):
    for idx, cluster in enumerate(list(cactus_partition.values())):
        # Sample fixed elements after clustering --> balanced dataset
        random_samples_number = min(20, len(cluster))
        cluster = sorted(random.sample(cluster, random_samples_number))
        cactus_partition[idx] = cluster
    return cactus_partition

def sample_random_data(cactus_partition):
    for idx, cluster in enumerate(list(cactus_partition.values())):
        # Sample random data from mini-Imagenet after clustering
        min_len = 10
        max_len = 30
        random_samples_number = random.randint(min_len, min(max_len, len(cluster)))
        cluster = sorted(random.sample(cluster, random_samples_number))
        cactus_partition[idx] = cluster
    return cactus_partition

def sample_unbalanced_data(cactus_partition):
    lens = np.asarray([len(el) for el in cactus_partition.values()])
    min, max = lens.min(), lens.max()
    new_min, new_max = 10, 30
    new_lens = []
    for cluster in cactus_partition.values():
        new_cluster_len = int((((new_max - new_min) * (len(cluster) - min)) / (max - min)) + new_min)
        new_lens.append(new_cluster_len)

    for idx, cluster_len in enumerate(new_lens):
        cactus_partition[idx] = sorted(random.sample(cactus_partition[idx], cluster_len))
    return cactus_partition

def sample_reducted_dataset(data, labels, num_classes):
    # Sample fixed random data from mini-Imagenet before clustering
    sample_elements = 20
    data = np.array_split(data, num_classes)
    labels = np.concatenate(np.asarray(np.split(labels, num_classes))[:, :sample_elements])
    new_classes = []
    for i, cls in enumerate(data):
        indices = np.random.choice(cls.shape[0], sample_elements, replace=False)
        new_cls = []
        for idx in indices:
            new_cls.append(cls[idx])
        new_classes.append(np.stack((new_cls)))
    new_classes = np.concatenate(new_classes)
    return new_classes, labels

def compute_weigth_vector(cactus_partition):
    min_len = 1000
    max_len = 0
    for el in cactus_partition.items():
        if len(el[1]) >= max_len:
            max_len = len(el[1])
        if len(el[1]) < min_len:
            min_len = len(el[1])

    max_key = max(cactus_partition.keys())
    empty = dict.fromkeys(range(max_key + 1), [])
    cactus_partition = {**empty, **cactus_partition}

    balance_vector = []
    for idx, el in enumerate(sorted(cactus_partition.items())):
        if len(el[1]) != min_len:
            balance_vector.append((max_len - min_len) / (len(el[1]) - min_len))
        elif len(el[1]) == 0:
            balance_vector.append(0)
        else:
            balance_vector.append((max_len - min_len) / ((len(el[1]) + 1) - min_len))

    balance_vector = np.asarray(balance_vector).astype(np.float32)
    balance_vector = (balance_vector - balance_vector.min()) / (balance_vector.max() - balance_vector.min())

    return balance_vector


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def remove_classes(trainset, to_keep):
    # trainset.data = trainset.data[order]
    trainset.targets = np.array(trainset.targets)
    # trainset.targets = trainset.targets[order]

    indices = np.zeros_like(trainset.targets)
    for a in to_keep:
        indices = indices + (trainset.targets == a).astype(int)
    indices = np.nonzero(indices)
    trainset.data = [trainset.data[i] for i in indices[0]]
    # trainset.data = trainset.data[indices]
    trainset.targets = np.array(trainset.targets)
    trainset.targets = trainset.targets[indices]

    return trainset
