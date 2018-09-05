import os
import os.path as osp

import chainer
import numpy as np
import scipy.misc
from sklearn.model_selection import train_test_split

import fcn
import fcn.image


class FoldingDataset(chainer.dataset.DatasetMixin):

    class_names = [
        'l_grasp',
        'r_grasp',
        'axis',
        'l_slide',
        'r_slide'

    ]
    mean_bgr = np.array((104.00698793, 116.66876762, 122.67891434))
    mean_d = np.array((127, 127, 127))
    mean_hand = np.array((127))

    def __init__(self, split, return_image=False, return_all=False):
        assert split in ('train', 'val')
        ids = self._get_ids()
        iter_train, iter_val = train_test_split(
            ids, test_size=0.2, random_state=np.random.RandomState(1234))
        self.ids = iter_train if split == 'train' else iter_val
        self._return_image = return_image
        self._return_all = return_all

    def __len__(self):
        return len(self.ids)

    def _get_ids(self):
        ids = []
        dataset_dir = chainer.dataset.get_dataset_directory(
            'sample/folding')
        for data_id in os.listdir(dataset_dir):
            ids.append(osp.join('folding', data_id))
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

    def get_example(self, i):
        ann_id, data_id = self.ids[i].split('/')
        assert ann_id in ('folding')

        dataset_dir = chainer.dataset.get_dataset_directory(
            'sample/folding_datasets')

        imgs = np.array(())
        for i in range(4):
            img_file_rgb = osp.join(dataset_dir, data_id, sorted(os.listdir('.'))[i],'image.png')
            img_rgb = scipy.misc.imread(img_file_rgb)
            datum_rgb = self.img_to_datum(img_rgb)
            imgs = np.vstack(imgs, datum_rgb)

        # imgs_d = np.array(())
        # for i in range(4):
        #     img_file_d = osp.join(dataset_dir, data_id, sorted(os.listdir('.'))[i], 'depth.npz')
        #     img_d = np.load(img_file_d)
        #     img_d_data = img_d['arr_0']
        #     img_d_jet = fcn.image.colorize_depth(img_d_data, 0.0, 2.0)
        #     datum_d = self.img_to_datum_d(img_d_jet)


        #datum = np.vstack((datum_rgb, datum_d))

        labels = np.array(())
        for i in range(4):
            label_file = osp.join(dataset_dir, data_id, sorted(os.listdir('.'))[i], 'label.png')
            label = scipy.misc.imread(label_file, mode='L')
            label = label.astype(np.int32)
            label[label == 255] = -1
            labels = np.vstack(labels, label)
        # if self._return_all:
        #     return datum, label, img_rgb, img_d_jet, img_hand[:, :, 0]
        # elif self._return_image:
        #     return datum, label, img_rgb
        # else:
        return imgs, labels


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = FoldingDataset('train', return_all=True)
    for i in range(len(dataset)):
        _, label, img_rgb, img_d_jet = dataset.get_example(i)
        labelviz = fcn.utils.label2rgb(label, img=img_rgb, label_names=dataset.class_names)
        plt.subplot(221)
        plt.imshow(img_rgb)
        plt.subplot(222)
        plt.imshow(img_d_jet)
        plt.subplot(223)
        plt.imshow(img_hand, cmap='gray')
        plt.subplot(224)
        plt.imshow(labelviz)
        plt.show()
