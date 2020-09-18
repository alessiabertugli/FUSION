import argparse
import logging
import numpy as np
import torch
import pathlib
from tensorboardX import SummaryWriter
import datasets.task_sampler as ts
import utils.utils as utils
from model.modelfactory import get_model
from utils.experiment import experiment
from model.meta_learner import MetaLearnerClassification
from model.learner import Learner
from datasets.utils import get_embeddings
from datasets.datasetfactory import get_dataset
from meta.meta_train import train, train_iid, test

logger = logging.getLogger('experiment')
LOG_DIR = pathlib.Path('path_to_logdir')


def dataset_handler(args):
    if args.dataset == "cifar":
        dataset = get_dataset(args, None, None, train=True, all=True)
        dataset_test = get_dataset(args, None, None, train=False, all=True)
    else:
        X_train, Y_train, Z_train, X_val, Y_val, Z_val = \
            get_embeddings(args.dataset, args.num_encoding_dims, args.test_set, args.encoder)

        if args.dataset == "imagenet":
            X_train, Y_train = utils.sample_reducted_dataset(X_train, Y_train, args.num_classes)

        elif args.dataset == "cub":
            indices_of_change = np.insert(np.where(Y_train[:-1] != Y_train[1:])[0] + 1, 0, 0)
            new_X_train = []
            for index in indices_of_change:
                new_X_train.extend(X_train[index:index + 20])  # sample 20 examples per class
            X_train = np.stack(new_X_train)
        Y_train = np.repeat(np.arange(100), 20)

        dataset = get_dataset(args, X_train, Y_train, train=True)
        dataset_test = get_dataset(args, X_train, Y_train, train=False)

    classes = np.unique(dataset.targets)
    total_classes_num = int(classes.shape[0] / 2)
    _, traj_classes = np.split(classes, np.argwhere(classes == classes[total_classes_num]).flatten())
    classes = list(classes)
    traj_classes = np.stack(classes)

    sampler = ts.SamplerFactory.get_sampler(args.dataset, classes, dataset, dataset_test)
    # Iterators used for evaluation
    iterator_train = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=args.iid, num_workers=4)
    iterator_test = torch.utils.data.DataLoader(dataset_test, batch_size=5, shuffle=False, num_workers=4)

    return traj_classes, sampler, iterator_train, iterator_test


def main(args):
    utils.set_seed(args.seed)

    my_experiment = experiment(args.name, args, LOG_DIR, commit_changes=args.commit)
    writer = SummaryWriter(my_experiment.path + "tensorboard")
    logger = logging.getLogger('experiment')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    config = get_model(args, args.num_classes)

    if not args.reload:
        if not args.iid:
            maml = MetaLearnerClassification(args, config, balance_vector=None).to(device)
        else:
            maml = Learner(config, args).to(device)
    else:
        maml = torch.load(args.ckpt_path).to(device)

    traj_classes, sampler, iterator_train, iterator_test = dataset_handler(args)
    opt = torch.optim.Adam(maml.parameters(), lr=args.update_lr)

    for step in range(args.steps):
        if args.iid:
            maml, accs, _ = train_iid(opt, maml, iterator_train, device)
        else:
            maml, accs, _ = train(args, traj_classes, sampler, maml)

        if args.iid and step == (args.steps-1):
                torch.save(maml, my_experiment.path + "learner.model")

        if step % 300 == 299:
            torch.save(maml.net, my_experiment.path + "learner.model")
            torch.save(maml, my_experiment.path + "meta-learner.model")

            correct = test(maml, args.iid, iterator_test, device)
            writer.add_scalar('/metatrain/test/classifier/accuracy', correct / len(iterator_test), step)
            logger.info("Test Accuracy = %s", str(correct / len(iterator_test)))

            correct_train_it = test(maml, args.iid, iterator_train, device)
            writer.add_scalar('/metatrain/train_iterator/classifier/accuracy', correct_train_it / len(iterator_train), step)
            logger.info("Train Iterator Accuracy = %s", str(correct_train_it / len(iterator_train)))

        # Evaluation during training for sanity checks
        if step % 40 == 39:
            if not args.iid:
                writer.add_scalar('/metatrain/train/accuracy', accs[-1], step)
            else:
                writer.add_scalar('/metatrain/train/accuracy', accs, step)
            logger.info('step: %d \t training acc %s', step, str(accs))

    writer.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--id_optim', type=int, default=None)
    argparser.add_argument('--steps', type=int, help='epoch number', default=40000)
    argparser.add_argument('--seed', type=int, help='Seed for random', default=10000)
    argparser.add_argument('--seeds', type=int, nargs='+', help='n way', default=[10])
    argparser.add_argument('--tasks', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--num_classes', type=int, help='number of classes', default=1100)
    argparser.add_argument('--name', help='Name of experiment', default="debug")
    argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")
    argparser.add_argument("--commit", action="store_true")
    argparser.add_argument("--no-reset", action="store_true")

    argparser.add_argument('--num_encoding_dims', type=int, help='of unsupervised representation learning method',
                           default=256)
    argparser.add_argument('--num_train_samples_per_class', type=int, help='number of examples per class for training',
                           default=15)
    argparser.add_argument('--encoder', type=str, help='acai or bigan or deepcluster or infogan',
                           default='acai')
    argparser.add_argument("--test_set", action="store_true",
                           help='Set to true to test on the the test set, False for the validation set.')
    argparser.add_argument('--reload', action="store_true")

    argparser.add_argument('--ckpt_path', type=str, default="", help="Checkpoint path")
    argparser.add_argument("--balancing", action="store_true")

    argparser.add_argument('--rehearsal', action="store_true")
    argparser.add_argument('--windows', type=int, help='windows', default=5)
    argparser.add_argument('--buffer_size', type=int, help='buffer size', default=10)

    argparser.add_argument('--attention', action="store_true")
    argparser.add_argument('--mean', action="store_true")

    argparser.add_argument('--iid', action="store_true")
    args = argparser.parse_args()

    args.name = "/".join([args.dataset, str(args.meta_lr).replace(".", "_"), args.name])
    print(args)
    main(args)