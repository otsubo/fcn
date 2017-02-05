import glob
import os
import os.path as osp
import re

import chainer
import numpy as np
import scipy.misc
from sklearn.model_selection import train_test_split
import skimage.filters
import skimage.color

import fcn


class JSKCableDataset(fcn.datasets.SegmentationDatasetBase):

    label_names = [
        'background',
        'cable',
        'plug',
    ]
    mean_bgr = np.array((104.00698793, 116.66876762, 122.67891434))

    def __init__(self, data_type):
        assert data_type in ('train', 'val')
        ids = self._get_ids()
        iter_train, iter_val = train_test_split(
            ids, test_size=0.2, random_state=np.random.RandomState(1234))
        self.ids = iter_train if data_type == 'train' else iter_val

    def __len__(self):
        return len(self.ids)

    def _get_ids(self):
        ids = []
        dataset_dir = chainer.dataset.get_dataset_directory('jsk_cable/JSKCableV1')
        for data_id in os.listdir(dataset_dir):
            ids.append(osp.join('cable', data_id))
        return ids

    def get_example(self, i):
        ann_id, data_id = self.ids[i].split('/')
        assert ann_id in ('cable')

        dataset_dir = chainer.dataset.get_dataset_directory('jsk_cable/JSKCableV1')

        img_file = osp.join(dataset_dir, data_id, 'img.png')
        img = scipy.misc.imread(img_file)
        datum = self.img_to_datum(img)

        label_file = osp.join(dataset_dir, data_id, 'label.png')
        label = scipy.misc.imread(label_file, mode='L')
        label[label == class_names.index('plug')] = 1
        prob = label.astype(np.float32)
        prob_filtered = skimage.filters.gaussian(prob, 5)
        prob_filtered[prob == 1] = 1.0
        # label[label == 255] = -1
        return datum, prob_filtered


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = JSKCableDataset('val')
    for i in xrange(len(dataset)):
        labelviz = dataset.visualize_example(i)
        plt.imshow(labelviz)
        plt.show()
