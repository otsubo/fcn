import json
import os
import os.path as osp

import chainer
import cv2

import imgaug
import imgaug.augmenters as iaa

import labelme
import mvtk
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
        'grasp',
        's_grasp',
        'v_grasp',
        'slide_target',
        'axis',
        'v_axis'

    ]
    mean_bgr = np.array((104.00698793, 116.66876762, 122.67891434))
    mean_d = np.array((127, 127, 127))
    mean_hand = np.array((127))
    video_dirs = []
    n_frames = []
    def __init__(self, split, return_image=False, return_all=False, img_viz=False, img_aug=False):
        assert split in ('train', 'val')
        self.split = split
        self.video_dirs = []
        self.n_frames = []
        self.video_dirs = sorted(self._get_video_dirs())
        video_dirs_train, video_dirs_val = train_test_split(
            self.video_dirs,
            test_size=0.2,
            random_state=np.random.RandomState(1234),
        )
        video_dirs_copy = []
        video_dirs_copy = video_dirs_train[:]
        for i in range(5):
            video_dirs_train.extend(video_dirs_copy)
        self.video_dirs = video_dirs_train if split == 'train' else video_dirs_val
        for video_dir in self.video_dirs:
            self.n_frames.append(len(os.listdir(video_dir)))
        self._return_image = return_image
        self._return_all = return_all
        self._img_viz = img_viz
        self._img_aug = img_aug

    def __len__(self):
        return len(self.video_dirs) * 4 
        #return sum(nf for vd, nf in self.video_dirs)
        #return sum(self.n_frames)

    def _get_video_dirs(self):
        self.vidoe_dirs = []
        video_dirs_copy = []
        d_path = '/home/otsubo/.chainer/dataset/folding_datasets_v7'
        dataset_dir = chainer.dataset.get_dataset_directory(
            'folding_datasets_v7')
        for video_dir in sorted(os.listdir(dataset_dir)):
            self.video_dirs.append(osp.join(d_path, video_dir))
        # if self.split == 'train':
        #     video_dirs_copy = self.video_dirs[:]
        #     for i in range(3):
        #         self.video_dirs.extend(video_dirs_copy)
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
        # seq = iaa.Sequential([
        #     iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
        #     iaa.Fliplr(0.5), # horizontally flip 50% of the images
        #     #iaa.GaussianBlur(sigma=(0, 3.0)), # blur images with a sigma of 0 to 3.0
        #     #iaa.ContrastNormalization((0.75, 1.5)),
        #     #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        #     #iaa.Multiply((0.8, 1.2), per_channel=0.2),
        #     iaa.Affine(
        #         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        #         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        #         rotate=(-25, 25),
        #         shear=(-8, 8)
        #     )
        # ],random_order=True)
        imgs = []
        imgs_raw = []
        lbl = []
        for frame_dir in sorted(os.listdir(video_dir))[:frame_index + 1]:
            frame = osp.join(video_dir, frame_dir)
            img_file = osp.join(frame, 'img_seg.png')
            img = scipy.misc.imread(img_file)
            imgs_raw.append(img)
            # if not self._img_viz:
            #     img = self.img_to_datum(img)
            imgs.append(img)
        for zero_img_index in range(self.n_frames[video_index] - (frame_index + 1)):
            img = np.zeros((hight, width, 3), np.uint8)
            imgs_raw.append(img)
            # if not self._img_viz:
            #     img = self.img_to_datum(img)
            imgs.append(img)
        #lbl_indices = frame_index
        json_file = osp.join(video_dir, (sorted(os.listdir(video_dir)))[frame_index], 'image.json')
        img_shape = (480, 640)
        lbl = self.json_file_to_lbl(img_shape, json_file)
        #print('lbl size {}'.format(lbl.shape))
        imgs = np.array(imgs)  # [(H, W, 3), (H, W, 3), ...]
        # if not self._img_viz:
        #     imgs = imgs.reshape(12, 480, 640)
        lbl = np.array(lbl)  # [(H, W), (H, W), ...]
        imgs_raw = np.array(imgs_raw)
        img_raw = imgs_raw[frame_index]

        if self._img_aug:
            imgs_tmp = []
            obj_datum = dict(img=imgs[i % 4,:,:,:])
            random_state = np.random.RandomState()
            st = lambda x: iaa.Sometimes(0.3,x)
            augs = [
                st(iaa.InColorspace(
                    'HSV', children=iaa.WithChannels([1, 2],
                                                     iaa.Multiply([0.5, 2])))),
                st(iaa.GaussianBlur(sigma=[0.0, 1.0])),
                st(iaa.AdditiveGaussianNoise(
                    scale=(0.0, 0.1 * 255), per_channel=True)),
            ]
            obj_datum = next(mvtk.aug.augment_object_data(
                [obj_datum], random_state=random_state, augmentations=augs))
            img_tmp = obj_datum['img']
            obj_datum2 = dict(img=img_tmp, lbl=lbl)
            random_state2 = np.random.RandomState()
            st2 = lambda x: iaa.Sometimes(0.7, x)  # NOQA
            augs2 = [
                st2(iaa.Affine(scale=(0.8, 1.2), order=0)),
                st2(iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})),
                st2(iaa.Affine(rotate=(-30, 30), order=0)),
                st2(iaa.Affine(shear=(-20, 20), order=0)),
            ]
            obj_datum2 = next(mvtk.aug.augment_object_data(
                [obj_datum2], random_state=random_state2, augmentations=augs2))
            imgs[i % 4,:,:,:] = obj_datum2['img']
            lbl = obj_datum2['lbl']
        if not self._img_viz:
            imgs_datum = []
            for j in range(4):
                img = imgs[j,:,:,:]
                img = self.img_to_datum(img)
                imgs_datum.append(img)
            imgs_datum = np.array(imgs_datum)
            imgs = imgs_datum
            imgs = imgs.reshape(12, 480, 640)

        if self._return_image:
            return imgs, lbl, img_raw
        else:
            return imgs, lbl




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = FoldingDataset('train', return_image=False,return_all=True, img_viz=True, img_aug=True)
    imgs_datum = []
    lbl_datum = []
    imgs = []
    lbl = []
    for i in range(sum(dataset.n_frames)):
        imgs, lbl  = dataset.get_example(i)
        imgs_datum.append(imgs)
        lbl_datum.append(lbl)
        print(i)
    imgs_datum = np.array(imgs_datum)
    #imgs_datum = imgs_datum.reshape(48, 4, 480, 640, 3)
    lbl_datum = np.array(lbl_datum)
    lbl_datum = lbl_datum.reshape(512, 480, 640)
    if dataset._img_viz:
        for i in range(512):
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
