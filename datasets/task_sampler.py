import copy
import logging
import numpy as np
import torch

logger = logging.getLogger("experiment")


class SamplerFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_sampler(dataset, tasks, trainset, testset=None):
        if "omni" in dataset:
            return OmniglotSampler(tasks, trainset, testset)
        elif "imagenet":
            return ImagenetSampler(tasks, trainset, testset)


class OmniglotSampler:
    # Class to sample tasks
    def __init__(self, tasks, trainset, testset):
        self.tasks = tasks
        self.task_sampler = SampleOmni(trainset, testset)
        self.task_sampler.add_complete_iterator(list(range(0, int(len(self.tasks)))))

    def get_complete_iterator(self):
        return self.task_sampler.complete_iterator

    def sample_task(self, t, train=True):
        return self.task_sampler.get(t, train)


class ImagenetSampler:
    # Class to sample tasks
    def __init__(self, tasks, trainset, testset):
        self.tasks = tasks
        self.task_sampler = SampleImagenet(trainset, testset)
        self.task_sampler.add_complete_iterator(list(range(0, int(len(self.tasks)))))

    def get_complete_iterator(self):
        return self.task_sampler.complete_iterator

    def sample_task(self, t, train=True):
        return self.task_sampler.get(t, train)

    def sample_tasks(self, t, train=False):
        # assert(false)
        dataset = self.task_sampler.get_task_trainset(t, train)
        train_iterator = torch.utils.data.DataLoader(dataset,
                                                     batch_size=1,
                                                     shuffle=True, num_workers=0)
        return train_iterator


class SampleOmni:

    def __init__(self, trainset, testset):
        self.task_iterators = []
        self.trainset = trainset
        self.testset = testset
        self.iterators = {}
        self.test_iterators = {}

    def add_complete_iterator(self, tasks):
        dataset = self.get_task_trainset(tasks, True)

        # dataset = self.get_task_testset(tasks)
        train_iterator = torch.utils.data.DataLoader(dataset,
                                                     batch_size=10,
                                                     shuffle=True, num_workers=0)
        self.complete_iterator = train_iterator
        logger.info("Len of complete iterator = %d", len(self.complete_iterator) * 256)

        train_iterator2 = torch.utils.data.DataLoader(dataset,
                                                      batch_size=1,
                                                      shuffle=True, num_workers=0)

        self.another_complete_iterator = train_iterator2

    def add_task_iterator(self, task, train):
        dataset = self.get_task_trainset([task], train)
        train_iterator = torch.utils.data.DataLoader(dataset,
                                                     batch_size=1,
                                                     shuffle=True, num_workers=0)
        self.iterators[task] = train_iterator
        print("Task %d has been added to the list" % task)
        return train_iterator

    def get(self, tasks, train):
        if train:
            for task in tasks:
                if task in self.iterators:
                    return self.iterators[task]
                else:
                    return self.add_task_iterator(task, True)
        else:
            for task in tasks:
                if tasks in self.test_iterators:
                    return self.test_iterators[task]
                else:
                    return self.add_task_iterator(task, False)

    def get_task_trainset(self, tasks, train):

        if train:
            set = copy.deepcopy(self.trainset)
        else:
            set = copy.deepcopy(self.testset)
        # class labels -> set.targets
        class_labels = np.asarray(set.targets)

        indices = np.zeros_like(class_labels)
        for a in tasks:
            indices = indices + (class_labels == a).astype(int)
        indices = np.nonzero(indices)

        set.data = [set.data[i] for i in indices[0]]
        set.targets = [set.targets[i] for i in indices[0]]

        set.data2 = []
        set.targets2 = []

        return set

    def filter_upto(self, task):

        trainset = copy.deepcopy(self.trainset)
        trainset.data = trainset.data[trainset.data['target'] <= task]

        return trainset


class SampleImagenet:

    def __init__(self, trainset, testset):
        self.task_iterators = []
        self.trainset = trainset
        self.testset = testset
        self.iterators = {}
        self.test_iterators = {}

    def add_complete_iterator(self, tasks):
        dataset = self.get_task_trainset(tasks, True)
        # dataset = self.get_task_testset(tasks)
        train_iterator = torch.utils.data.DataLoader(dataset,
                                                     batch_size=10,
                                                     shuffle=True, num_workers=0)
        self.complete_iterator = train_iterator
        logger.info("Len of complete iterator = %d", len(self.complete_iterator) * 256)

        train_iterator2 = torch.utils.data.DataLoader(dataset,
                                                      batch_size=1,
                                                      shuffle=True, num_workers=0)

        self.another_complete_iterator = train_iterator2

    def add_task_iterator(self, task, train):

        dataset = self.get_task_trainset([task], train)
        train_iterator = torch.utils.data.DataLoader(dataset,
                                                     batch_size=1,
                                                     shuffle=True, num_workers=0)
        self.iterators[task] = train_iterator
        print("Task %d has been added to the list" % task)
        return train_iterator

    def get(self, tasks, train):
        if train:
            for task in tasks:
                if task in self.iterators:
                    return self.iterators[task]
                else:
                    return self.add_task_iterator(task, True)
        else:
            for task in tasks:
                if tasks in self.test_iterators:
                    return self.test_iterators[task]
                else:
                    return self.add_task_iterator(task, False)

    def get_task_trainset(self, tasks, train):

        if train:
            set = copy.deepcopy(self.trainset)
        else:
            set = copy.deepcopy(self.testset)
        # class labels -> set.targets
        class_labels = np.asarray(set.targets)

        indices = np.zeros_like(class_labels)
        for a in tasks:
            indices = indices + (class_labels == a).astype(int)
        indices = np.nonzero(indices)

        set.data = [set.data[i] for i in indices[0]]
        set.targets = [set.targets[i] for i in indices[0]]

        set.data2 = []
        set.targets2 = []

        return set

    def filter_upto(self, task):

        trainset = copy.deepcopy(self.trainset)
        trainset.data = trainset.data[trainset.data['target'] <= task]

        return trainset

