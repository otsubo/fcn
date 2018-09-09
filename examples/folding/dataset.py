import json
import os
import os.path as osp

import chainer
import labelme
import numpy as np
import scipy.misc
from sklearn.model_selection import train_test_split

import fcn
import fcn.image


class FoldingDataset(chainer.dataset.DatasetMixin):

    class_names = [
        '__background__',
        'l_grasp',
        'r_grasp',
        'axis',
        'l_slide',
        'r_slide'

    ]
    mean_bgr = np.array((104.00698793, 116.66876762, 122.67891434))
    mean_d = np.array((127, 127, 127))
    mean_hand = np.array((127))

    def __init__(self, split, return_image=False, return_all=False, img_viz=False):
        assert split in ('train', 'val')
        video_dirs = self._get_video_dirs()
        video_dirs_train, video_dirs_val = train_test_split(
            video_dirs,
            test_size=0.2,
            random_state=np.random.RandomState(1234),
        )
        self.video_dirs = video_dirs_train if split == 'train' else video_dirs_val
        self._return_image = return_image
        self._return_all = return_all
        self._img_viz = img_viz

    def __len__(self):
        return len(self.video_dirs)

    def _get_video_dirs(self):
        # dataset_dir = chainer.dataset.get_dataset_directory(
        #     'sample/folding')
        import glob
        dataset_dir = '/home/otsubo/.ros/jsk_data/sample/folding_datasets'
        video_dirs = glob.glob(osp.join(dataset_dir, 'folding*'))
        return video_dirs 

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

    def json_file_to_lbl(self, img_shape, json_file):
        label_name_to_value = {}
        for label_value, label_name in enumerate(self.class_names):
            label_name_to_value[label_name] = label_value
        with open(json_file) as f:
            data = json.load(f)
        lbl = labelme.utils.shapes_to_label(
            img_shape, data['shapes'], label_name_to_value
        )
        return lbl

    def get_example(self, i):
        video_dir = self.video_dirs[i]

        imgs = []
        lbls = []
        for frame_dir in sorted(os.listdir(video_dir)):
            frame_dir = osp.join(video_dir, frame_dir)
            img_file = osp.join(frame_dir, 'image.png')
            img = scipy.misc.imread(img_file)

            json_file = osp.join(frame_dir, 'image.json')
            lbl = self.json_file_to_lbl(img.shape, json_file)
            if (not self._img_viz):
                img = self.img_to_datum(img)
            imgs.append(img)  # (H, W, 3)
            lbls.append(lbl)
        imgs = np.array(imgs)  # [(H, W, 3), (H, W, 3), ...]
        lbls = np.array(lbls)  # [(H, W), (H, W), ...]
        #print(lbls.shape)

        N, C, H, W = imgs.shape
        assert N == 4
        assert C == 3
        assert lbls.shape == (N, H, W)

        imgs = imgs.transpose(0, 3, 1, 2).reshape(N * C, H, W)  # L.Convolution2D(12, 64, ksize=3, stride=1, pad=1)

        assert imgs.shape == (N * C, H, W)
        assert lbls.shape == (N, H, W)

        return imgs, lbls

        # print(video_dir)
        # quit()

        # ann_id, data_id = self.ids[i].split('/')
        # assert ann_id in ('folding')

        # dataset_dir = chainer.dataset.get_dataset_directory(
        #     'sample/folding_datasets')

        # imgs = np.array(())
        # for i in range(4):
        #     img_file_rgb = osp.join(dataset_dir, data_id, sorted(os.listdir('.'))[i],'image.png')
        #     img_rgb = scipy.misc.imread(img_file_rgb)
        #     datum_rgb = self.img_to_datum(img_rgb)
        #     imgs = np.vstack(imgs, datum_rgb)

        # # imgs_d = np.array(())
        # # for i in range(4):
        # #     img_file_d = osp.join(dataset_dir, data_id, sorted(os.listdir('.'))[i], 'depth.npz')
        # #     img_d = np.load(img_file_d)
        # #     img_d_data = img_d['arr_0']
        # #     img_d_jet = fcn.image.colorize_depth(img_d_data, 0.0, 2.0)
        # #     datum_d = self.img_to_datum_d(img_d_jet)


        # #datum = np.vstack((datum_rgb, datum_d))

        # labels = np.array(())
        # for i in range(4):
        #     label_file = osp.join(dataset_dir, data_id, sorted(os.listdir('.'))[i], 'label.png')
        #     label = scipy.misc.imread(label_file, mode='L')
        #     label = label.astype(np.int32)
        #     label[label == 255] = -1
        #     labels = np.vstack(labels, label)
        # # if self._return_all:
        # #     return datum, label, img_rgb, img_d_jet, img_hand[:, :, 0]
        # # elif self._return_image:
        # #     return datum, label, img_rgb
        # # else:
        # return imgs, labels


if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    dataset = FoldingDataset('train', return_all=True)
    for i in range(len(dataset)):
        imgs, lbls = dataset.get_example(i)
        NxC, H, W = imgs.shape
        C = 3
        N = NxC // C
        imgs = imgs.reshape(N, C, H, W)
        imgs = imgs.transpose(0, 2, 3, 1)

        assert imgs.shape == (N, H, W, C)
        assert lbls.shape == (N, H, W)
        if dataset._img_viz :
            for img, lbl in zip(imgs, lbls):
                viz = fcn.utils.label2rgb(lbl, img, label_names=dataset.class_names)
                viz = np.hstack((img, viz))
                cv2.imshow(__file__, viz[:, :, ::-1])
                if cv2.waitKey(0) == ord('q'):
                    quit()
        # _, label, img_rgb, img_d_jet = dataset.get_example(i)
        # labelviz = fcn.utils.label2rgb(label, img=img_rgb, label_names=dataset.class_names)
        # plt.subplot(221)
        # plt.imshow(img_rgb)
        # plt.subplot(222)
        # plt.imshow(img_d_jet)
        # plt.subplot(223)
        # plt.imshow(img_hand, cmap='gray')
        # plt.subplot(224)
        # plt.imshow(labelviz)
        # plt.show()
