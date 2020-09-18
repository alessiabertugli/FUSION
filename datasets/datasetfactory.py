import datasets.omniglot as om
import datasets.miniimagenet as mi
import datasets.cifar as cf
import datasets.cub as cub
from datasets.task_generator import TaskGenerator


def get_dataset(args, data, labels, train=True, all=False):
    if args.dataset == "omniglot":
        return om.Omniglot(data, labels, train=train, all=all)
    elif args.dataset == "imagenet":
        return mi.MiniImagenet(data, labels, train=train, all=all)
    elif args.dataset == "cifar":
        return cf.Cifar(train=train, all=all)
    elif args.dataset == "cub":
        return cub.Cub(data, labels, train=train, all=all)
    else:
        print("Unsupported Dataset")
        assert False


def cactus(args, X, Z, Y, train):
    num_samples_per_class = args.num_train_samples_per_class + args.num_val_samples_per_class
    task_generator = TaskGenerator(num_samples_per_class=num_samples_per_class, args=args)
    partition = task_generator.get_partitions_kmeans(encodings=Z, train=train)
    data, labels, true_labels = task_generator.make_unsupervised_dataset(data=X, partition=partition, true_labels=Y)
    return data, labels, true_labels


def get_dataset_unbalanced(args, data, partition, train=True, all=False):
    if args.dataset == "omniglot":
        if not args.aug:
            return om.OmniglotUnbalanced(data, partition, train=train, all=all)
        else:
            return om.OmniglotAugmentation(data, partition, train=train, all=all)
    elif args.dataset == "imagenet":
        return mi.MiniImagenetUnbalanced(data, partition, train=train, all=all)
    elif args.dataset == "cub":
        return cub.CubUnsupervised(data, partition, train=train, all=all)
    else:
        print("Unsupported Dataset")
        assert False


def cactus_unbalanced(args, Z, train):
    num_samples_per_class = args.lb
    task_generator = TaskGenerator(num_samples_per_class=num_samples_per_class, args=args)
    partition = task_generator.get_partitions_kmeans(encodings=Z, train=train)
    return partition
