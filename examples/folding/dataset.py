import json
import os
import os.path as osp

import chainer
import cv2

import labelme
import numpy as np
import scipy.misc
from sklearn.model_selection import train_test_split

import fcn
import fcn.image

width = 640
hight = 480

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
    video_dirs = []
    n_frames = []
    videos = []
    def __init__(self, split, return_image=False, return_all=False, img_viz=False):
        assert split in ('train', 'val')
        self.video_dirs = sorted(self._get_video_dirs())
        video_dirs_train, video_dirs_val = train_test_split(
            self.video_dirs,
            test_size=0.2,
            random_state=np.random.RandomState(1234),
        )
        self.video_dirs = video_dirs_train if split == 'train' else video_dirs_val
        for video_dir in self.video_dirs:
            self.n_frames.append(len(os.listdir(video_dir)))
        self._return_image = return_image
        self._return_all = return_all
        self._img_viz = img_viz

    def __len__(self):
        return len(self.video_dirs) * 4
        #return sum(nf for vd, nf in self.video_dirs)
        #return sum(self.n_frames)

    def _get_video_dirs(self):
        self.vidoe_dirs = []
        d_path = '/home/otsubo/.chainer/dataset/folding_datasets'
        #self.video_dirs = glob.glob(osp.join(dataset_dir, 'folding*'))
        #self.video_dirs = glob.glob(osp.join(dataset_dir))
        dataset_dir = chainer.dataset.get_dataset_directory(
            'folding_datasets')
        for video_dir in os.listdir(dataset_dir):
            self.video_dirs.append(osp.join(d_path, video_dir))
        #self.video_dirs = osp.join(d_path ,os.listdir(dataset_dir))
        return self.video_dirs

    def img_to_datum(self, img):
        img = img.copy()
        datum = img.astype(np.float32)
        datum = datum[:, :, ::-1]  # RGB -> BGR
        datum -= self.mean_bgr
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
        video_index = i // self.n_frames[0]
        frame_index = i % self.n_frames[0]
        video_dir = self.video_dirs[video_index]
        imgs = []
        imgs_raw = []
        lbl = []
        for frame_dir in sorted(os.listdir(video_dir))[:frame_index + 1]:
            frame = osp.join(video_dir, frame_dir)
            img_file = osp.join(frame, 'image.png')
            img = scipy.misc.imread(img_file)
            imgs_raw.append(img)
            img = self.img_to_datum(img)
            imgs.append(img)
        for zero_img_index in range(self.n_frames[video_index] - (frame_index + 1)):
            img = np.zeros((hight, width, 3), np.uint8)
            imgs_raw.append(img)
            img = self.img_to_datum(img)
            imgs.append(img)
        #lbl_indices = frame_index
        json_file = osp.join(video_dir, (sorted(os.listdir(video_dir)))[frame_index], 'image.json')
        img_shape = img.shape[1:3]
        #img_shape = (480, 640)
        lbl = self.json_file_to_lbl(img_shape, json_file)

        imgs = np.array(imgs)  # [(H, W, 3), (H, W, 3), ...]
        imgs = imgs.reshape(12, 480, 640)
        lbl = np.array(lbl)  # [(H, W), (H, W), ...]
        imgs_raw = np.array(imgs_raw)
        img_raw = imgs_raw[frame_index]
        if self._return_image:
            return imgs, lbl, img_raw
        else:
            return imgs, lbl




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = FoldingDataset('train', return_all=True, img_viz=False)
    dataset_val = FoldingDataset('val', return_all=True, img_viz=False)
    print('train data video_dirs {}\n'.format(dataset.video_dirs))
    print('val data video_dirs {}\n'.format(dataset_val.video_dirs))
    imgs_datum = []
    lbl_datum = []
    for i in range(sum(dataset.n_frames)):
        imgs, lbl  = dataset.get_example(i)
        imgs_datum.append(imgs)
        lbl_datum.append(lbl)
        #lbl_indices_datum.append(lbl_indices)
    imgs_datum = np.array(imgs_datum)
    imgs_datum = imgs_datum.reshape(48, 4, 480, 640, 3)
    lbl_datum = np.array(lbl_datum)
    lbl_datum = lbl_datum.reshape(48, 480, 640)
    # NxC, H, W = imgs.shape
    # C = 3
    # N = NxC // C
    #imgs = imgs.reshape(N, C, H, W)
    #imgs = imgs.transpose(0, 2, 3, 1)
    #import ipdb; ipdb.set_trace()
    #assert imgs.shape == (N, H, W, C)
    #assert lbls.shape == (N, H, W)
    # if dataset._img_viz :
    #     for img, lbl in zip(imgs, lbl):
    #         viz = fcn.utils.label2rgb(lbl, img, label_names=dataset.class_names)
    #         viz = np.hstack((img, viz))
    #         cv2.imshow(__file__, viz[:, :, ::-1])
    #         if cv2.waitKey(0) == ord('q'):
    #             quit()
    if dataset._img_viz:
    #import ipdb; ipdb.set_trace()
        for i in range(48):
            img = imgs_datum[i,:,:,:][0,:,:,:]
            img2 = imgs_datum[i,:,:,:][1,:,:,:]
            img3 = imgs_datum[i,:,:,:][2,:,:,:]
            img4 = imgs_datum[i,:,:,:][3,:,:,:]
            lbl = lbl_datum[i,:,:]
            viz_img = imgs_datum[i,:,:,:][i % 4,:,:,:]
            labelviz = fcn.utils.label2rgb(lbl, img=viz_img, label_names=dataset.class_names)
            viz = fcn.utils.get_tile_image([img, img2, img3, img4, labelviz])
            cv2.imshow(__file__, viz[:, :, ::-1])
            if cv2.waitKey(0) == ord('q'):
                quit()
