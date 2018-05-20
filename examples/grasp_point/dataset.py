import os
import os.path as osp

import chainer
import numpy as np
import scipy.misc
from sklearn.model_selection import train_test_split

import fcn
import fcn.image


class GraspPointDataset(chainer.dataset.DatasetMixin):

    class_names = [
        'background',
        'grasp_point'
    ]
    mean_bgr = np.array((104.00698793, 116.66876762, 122.67891434))
    mean_d = np.array((127, 127, 127))
    mean_hand = np.array((127))

    def __init__(self, split, return_image=False):
        assert split in ('train', 'val')
        ids = self._get_ids()
        iter_train, iter_val = train_test_split(
            ids, test_size=0.2, random_state=np.random.RandomState(1234))
        self.ids = iter_train if split == 'train' else iter_val
        self._return_image = return_image

    def __len__(self):
        return len(self.ids)

    def _get_ids(self):
        ids = []
        dataset_dir = chainer.dataset.get_dataset_directory(
            'grasp_point/GraspPointV1')
        for data_id in os.listdir(dataset_dir):
            ids.append(osp.join('grasp_point', data_id))
        return ids

    def img_to_datum(self, img):
        img = img.copy()
        datum = img.astype(np.float32)
        datum = datum[:, :, ::-1]  # RGB -> BGR
        datum -= self.mean_bgr
        datum = datum.transpose((2, 0, 1))
        return datum

    def img_to_datum_d(self, img):
        img = img.copy()
        datum = img.astype(np.float32)
        datum -= self.mean_d
        datum = datum.transpose((2, 0, 1))
        return datum

    def img_to_datum_hand(self, img):
        img = img.copy()
        datum = img.astype(np.float32)
        datum -= self.mean_hand
        datum = datum.transpose((2, 0, 1))
        return datum

    def get_example(self, i):
        ann_id, data_id = self.ids[i].split('/')
        assert ann_id in ('grasp_point')

        dataset_dir = chainer.dataset.get_dataset_directory(
            'grasp_point/GraspPointV1')

        img_file_rgb = osp.join(dataset_dir, data_id, 'img_rgb.png')
        img_rgb = scipy.misc.imread(img_file_rgb)
        datum_rgb = self.img_to_datum(img_rgb)

        img_file_d = osp.join(dataset_dir, data_id, 'img_d.npz')
        img_d = np.load(img_file_d)
        img_d_data = img_d['arr_0']
        img_d_jet = fcn.image.colorize_depth(img_d_data, 0.0, 2.0)
        datum_d = self.img_to_datum_d(img_d_jet)

        img_file_hand = osp.join(dataset_dir, data_id, 'img_hand.png')
        img_hand = scipy.misc.imread(img_file_hand)
        img_hand = img_hand.reshape(544, 1024, 1)
        datum_hand = self.img_to_datum_hand(img_hand)

        datum = np.vstack((datum_rgb, datum_d, datum_hand))

        label_file = osp.join(dataset_dir, data_id, 'label.png')
        label = scipy.misc.imread(label_file, mode='L')
        label = label.astype(np.int32)
        label[label == 255] = -1
        if self._return_image:
            return datum, label, img_rgb
        return datum, label


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     dataset = GraspPointDataset('val')
#     for i in xrange(len(dataset)):
#         datum, label = dataset.get_example(i)
#         plt.imshow(labelviz)
#         plt.show()
