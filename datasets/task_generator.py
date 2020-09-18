import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
import os
from sklearn.cluster import KMeans

os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'  # default runs out of space for parallel processing


class TaskGenerator(object):
    def __init__(self, num_samples_per_class, args):
        self.num_samples_per_class = num_samples_per_class
        self.args = args

    def make_unsupervised_dataset(self, data, partition, true_labels):
        """
        Make unsupervised dataset associating the predicted labels to the corresponding images, sampling the fixed
        number of elements per class and ordering data per labels
        """
        new_labels = [-1] * len(data)
        for idx, cluster in enumerate(list(partition.values())):
            if len(cluster) > self.num_samples_per_class:
                cluster = sorted(random.sample(cluster, self.num_samples_per_class))
            for img in cluster:
                new_labels[img] = list(partition.keys())[idx]

        empty_indices = np.argwhere(np.asarray(new_labels) == -1).flatten()
        new_data = np.delete(data, empty_indices, axis=0)
        new_true_labels = np.delete(true_labels, empty_indices, axis=0)
        new_labels = np.asarray(new_labels)
        new_labels = new_labels[new_labels != -1]
        sorted_indices = np.argsort(new_labels)
        sorted_labels = np.sort(new_labels)
        sorted_data = []
        sorted_true_labels = []
        for index in sorted_indices:
            sorted_data.append(new_data[index])
            sorted_true_labels.append(new_true_labels[index])
        sorted_data = np.asarray(sorted_data)
        sorted_true_labels = np.asarray(sorted_true_labels)
        return sorted_data, sorted_labels, sorted_true_labels

    def get_partitions_kmeans(self, encodings, train):
        encodings_list = [encodings]
        if train:
            if self.args.scaled_encodings:
                n_clusters_list = [self.args.num_clusters]
                for i in range(self.args.num_partitions - 1):
                    weight_vector = np.random.uniform(low=0.0, high=1.0, size=encodings.shape[1])
                    encodings_list.append(np.multiply(encodings, weight_vector))
            else:
                n_clusters_list = [self.args.num_clusters] * self.args.num_partitions
        else:
            n_clusters_list = [self.args.num_clusters]
        assert len(encodings_list) * len(n_clusters_list) == self.args.num_partitions
        if self.args.num_partitions != 1:
            n_init = 1  # so it doesn't take forever
        else:
            n_init = 10
        init = 'k-means++'

        print('Number of encodings: {}, number of n_clusters: {}, number of inits: '.format(len(encodings_list), len(n_clusters_list)), n_init)

        kmeans_list = []
        for n_clusters in tqdm(n_clusters_list, desc='get_partitions_kmeans_n_clusters'):
            for encodings in tqdm(encodings_list, desc='get_partitions_kmeans_encodings'):
                while True:
                    kmeans = KMeans(n_clusters=n_clusters, init=init, precompute_distances=True, random_state=128, n_jobs=40,
                                    n_init=n_init, max_iter=3000).fit(encodings)
                    uniques, counts = np.unique(kmeans.labels_, return_counts=True)
                    num_big_enough_clusters = np.sum(counts > self.num_samples_per_class)
                    if num_big_enough_clusters > 0.30 * n_clusters:  #0.75
                        break
                    else:
                        tqdm.write("Too few classes ({}) with greater than {} examples.".format(num_big_enough_clusters,
                                                                                           self.num_samples_per_class))
                        tqdm.write('Frequency: {}'.format(counts))
                kmeans_list.append(kmeans)

        partition = self.get_partition_from_labels(kmeans_list[-1].labels_)
        return partition

    def get_partition_from_labels(self, labels):
        """
        Constructs the partition of the set of indices in labels, grouping indices according to their label.
        :param labels: np.array of labels, whose i-th element is the label for the i-th datapoint
        :return: a dictionary mapping class label to a list of indices that have that label
        """
        partition = defaultdict(list)
        for ind, label in enumerate(labels):
            partition[label].append(ind)
        if not self.args.aug:
            self.clean_partition(partition)
        return partition

    def clean_partition(self, partition):
        """
        Removes subsets that are too small from a partition.
        """
        for cls in list(partition.keys()):
            if len(partition[cls]) < self.num_samples_per_class:
                del(partition[cls])
        return partition