import argparse
import logging
import random
import torch
from tensorboardX import SummaryWriter
import pathlib
import numpy as np
from utils.utils import remove_classes, sample_reducted_dataset
from datasets.utils import get_embeddings
from datasets.datasetfactory import get_dataset
from utils.experiment import experiment
from meta.meta_test import lr_search, train, test, model_loader

logger = logging.getLogger('experiment')
LOG_DIR = pathlib.Path('path_to_logdir')


def dataset_handler(args, tot_class):
    if args.dataset == "omniglot" or args.dataset == "imagenet" or args.dataset == "cub":
        X_train, Y_train, Z_train, X_test, Y_test, Z_test = \
            get_embeddings(args.dataset, args.num_encoding_dims, args.test_set, args.encoder)
        Y_test = Y_test - np.min(Y_test)
        classes = np.max(Y_test) + 1

        if args.dataset == "imagenet":
            X_test, Y_test = sample_reducted_dataset(X_test, Y_test, classes)

        if args.dataset == "cub":
            indices_of_change = np.insert(np.where(Y_test[:-1] != Y_test[1:])[0] + 1, 0, 0)
            new_X_test, new_Y_test = [], []
            for index in indices_of_change:
                new_X_test.extend(X_test[index:index+20])  #sample 20 examples per class
                new_Y_test.extend(Y_test[index:index+20])
            X_test, Y_test = np.stack(new_X_test), np.stack(new_Y_test)
            Y_test = np.repeat(np.arange(len(np.unique(np.stack(new_Y_test)))), 20)
            classes = np.max(Y_test) + 1

        keep = np.random.choice(list(range(classes)), tot_class, replace=False)
        dataset = remove_classes(get_dataset(args, X_test, Y_test, train=True), keep)
        iterator_sorted = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=args.iid, num_workers=0)
        dataset = remove_classes(get_dataset(args, X_test, Y_test, train=False), keep)
        iterator = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    else:
        dataset = get_dataset(args, None, None, True, all=False)
        classes = np.max(dataset.targets) + 1
        keep = np.random.choice(list(range(classes)), tot_class, replace=False)
        dataset = remove_classes(get_dataset(args, None, None, train=True, all=False), keep)
        iterator_sorted = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=args.iid, num_workers=0)
        dataset = remove_classes(get_dataset(args, None, None, train=False, all=False), keep)
        iterator = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    return iterator, iterator_sorted, classes


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    my_experiment = experiment(args.name, args, LOG_DIR, args.commit)
    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')
    logger.setLevel(logging.INFO)

    # Decrease learning rate at these epochs
    if args.dataset == "omniglot":
        schedule = [10, 50, 75, 100, 150, 200]
    elif args.dataset == "imagenet" or args.dataset == "cifar":
        schedule = [2, 4, 6, 8, 10]
    elif args.dataset == "cub":
        schedule = [2, 10, 20, 30, 40]
    else:
        print("Unsupported dataset")
        assert (False)

    if args.attention or args.mean:
        aggregation = True
    else:
        aggregation = False

    final_results_all = []

    for tot_class in schedule:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        iterator, iterator_sorted, classes = dataset_handler(args, tot_class)
        # Learning rate search
        best_lr = lr_search(args, classes, iterator_sorted, iterator, logger, aggregation, device)

        for aoo in range(0, args.runs):
            maml = model_loader(args, classes, device)
            # Meta-test training phase
            maml = train(args, maml, iterator_sorted, aggregation, device, best_lr)
            # Meta-test test phase
            current_acc = test(logger, maml, iterator, aggregation, device)

            logger.info("Final Max Result = %s", str(current_acc))
            writer.add_scalar('/finetune/best_' + str(aoo), current_acc, tot_class)
            final_results_all.append((tot_class, current_acc))
            print("A=  ", current_acc)
            logger.info("Final results = %s", str(current_acc))

            my_experiment.results["Final Results"] = final_results_all
            my_experiment.store_json()
            print("FINAL RESULTS = ", final_results_all)

            # mean and std of the results
            accs_current_cls = np.array([res[1] for res in final_results_all if res[0] == tot_class])
            logger.info("Task %d, mean: %s, std: %s", tot_class, str(np.mean(accs_current_cls)),
                        str(np.std(accs_current_cls)))
    writer.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--id_optim', type=int, default=None)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1)
    argparser.add_argument('--seed', type=int, help='epoch number', default=222)
    argparser.add_argument('--model', type=str, help='epoch number',
                           default="path_to_model")
    argparser.add_argument('--scratch', action='store_true')
    argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")
    argparser.add_argument('--name', help='Name of experiment', default="evaluation")
    argparser.add_argument("--commit", action="store_true")
    argparser.add_argument("--no-freeze", action="store_true")
    argparser.add_argument('--reset', action="store_true")
    argparser.add_argument("--iid", action="store_true")
    argparser.add_argument("--runs", type=int, default=50)

    argparser.add_argument('--num_encoding_dims', type=int, help='of unsupervised representation learning method', default=256)
    argparser.add_argument('--encoder', type=str, help='acai or bigan or deepcluster or infogan', default='acai')
    argparser.add_argument("--test_set", action="store_false", help='Set to true to test on the the test set, False for the validation set.')
    argparser.add_argument('--num_train_samples_per_class', type=int, help='number of examples per class for training',
                           default=15)

    argparser.add_argument('--rehearsal', action="store_true")
    argparser.add_argument('--windows', type=int, help='windows', default=5)
    argparser.add_argument('--buffer_size', type=int, help='buffer size', default=10)

    argparser.add_argument('--attention', action="store_true")
    argparser.add_argument('--mean', action="store_true")

    args = argparser.parse_args()

    args.name = "/".join([args.dataset, "eval", str(args.epoch).replace(".", "_"), args.name])
    args = set_args_test(args)
    print (args)
    main(args)