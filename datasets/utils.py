import errno
import hashlib
import os
import os.path
import numpy as np
from torch.utils.model_zoo import tqdm


def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def check_integrity(fpath, md5=None):
    if md5 is None:
        return True
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater()
            )
        except OSError:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater()
                )


def list_dir(root, prefix=False):
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def get_embeddings(dataset, num_encoding_dims, test_set, encoder):
    """
    Assumes each split of a dataset is in a .npz file with keys 'X', 'Y', and 'Z' corresponding to uint8 images, labels, and float32 encodings, respectively.
    """
    splits = ['train', 'val', 'test']
    print('Encoder: {}'.format(encoder))

    if dataset == "omniglot":
        if encoder == 'bigan':
            data_folder = 'path_to_bigan_encodings'
            filenames = {split: os.path.join(data_folder, '{}.u-{}_{}.npz'.format(dataset, num_encoding_dims, split))
                         for split in splits}
        elif encoder == 'acai':
            data_folder = 'path_to_acai_encodings'
            filenames = {split: os.path.join(data_folder, '{}_{}_{}.npz'.format(dataset, num_encoding_dims, split))
                         for split in splits}
        elif encoder == 'sela_resnetv2':
            print('SeLa embeddings are whitened and normalized already!')
            data_folder = 'path_to_sela_encodings'
            filenames = {split: os.path.join(data_folder, '{}_{}_{}.npz'.format(dataset, "resnetv2_84", split))
                         for split in splits}
        elif encoder == 'sela_alexnet':
            print('SeLa embeddings are whitened and normalized already!')
            data_folder = 'path_to_sela_encodings'
            filenames = {split: os.path.join(data_folder, '{}_{}_{}.npz'.format(dataset, "alexnet", split))
                         for split in splits}
    elif dataset == "imagenet":
        print('Deep cluster embeddings are whitened and normalized already!')
        data_folder = 'path_to_deepcluster_encodings'
        filenames = {split: os.path.join(data_folder, '{}_{}_{}.npz'.format("miniimagenet", num_encoding_dims, split))
                     for split in splits}
    elif dataset == "cub":
        data_folder = 'path_to_ood_data/cub/'

        X_train, Y_train = np.load(data_folder + "imgs_train_cub.npy"), np.load(data_folder + "labels_train_cub.npy")
        X_test, Y_test = np.load(data_folder + "imgs_test_cub.npy"), np.load(data_folder + "labels_test_cub.npy")
        Z_train, Z_test = None, None
    else:
        raise NotImplementedError

    def get_XYZ(filename):
        data = np.load(filename)
        if encoder == 'infogan':
            X, Y, Z = data['X'], data['Y'], data['Z_raw']
        else:
            X, Y, Z = data['X'], data['Y'], data['Z']
        return X, Y, Z

    if dataset == "omniglot" or dataset == "imagenet":
        X_train, Y_train, Z_train = get_XYZ(filenames['train'])
        X_val, Y_val, Z_val = get_XYZ(filenames['val'])
        X_test, Y_test, Z_test = get_XYZ(filenames['test'])

        if  encoder == 'sela_alexnet' or encoder == 'sela_resnetv2':
            X_train = X_train.transpose(0, 2, 3, 1)
            X_test = X_test.transpose(0, 2, 3, 1)
            X_val = X_val.transpose(0, 2, 3, 1)

        if not test_set:  # use val as test
            X_test, Y_test, Z_test = X_val, Y_val, Z_val

        return X_train, Y_train, Z_train, X_test, Y_test, Z_test
    else:
        return X_train, Y_train, Z_train, X_test, Y_test, Z_test
