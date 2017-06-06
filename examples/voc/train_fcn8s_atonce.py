#!/usr/bin/env python

import datetime
import os.path as osp
import subprocess

import chainer
from chainer import cuda
import click

import fcn
from fcn import datasets


here = osp.dirname(osp.abspath(__file__))


@click.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option('-g', '--gpu', type=int, required=True)
def main(gpu):
    # 0. config

    cmd = 'git log -n1 --format="%h"'
    vcs_version = subprocess.check_output(cmd, shell=True).strip()
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    out = 'fcn8s_atonce_VCS-%s_TIME-%s' % (
        vcs_version,
        timestamp,
    )
    out = osp.join(here, 'logs', out)

    # 1. dataset

    dataset_train = datasets.SBDClassSeg(split='train')
    dataset_valid = datasets.VOC2011ClassSeg(split='seg11valid')

    iter_train = chainer.iterators.MultiprocessIterator(
        dataset_train, batch_size=1, shared_mem=10**7)
    iter_valid = chainer.iterators.MultiprocessIterator(
        dataset_valid, batch_size=1, shared_mem=10**7,
        repeat=False, shuffle=False)

    # 2. model

    n_class = len(dataset_train.class_names)

    vgg_path = fcn.data.download_vgg16_chainermodel(check_md5=False)
    vgg = fcn.models.VGG16()
    chainer.serializers.load_npz(vgg_path, vgg)

    model = fcn.models.FCN8sAtOnce(n_class=n_class)
    model.init_from_vgg16(vgg)

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    # 3. optimizer

    optimizer = chainer.optimizers.MomentumSGD(lr=1.0e-10, momentum=0.99)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))
    for p in model.params():
        if p.name == 'b':
            p.update_rule = chainer.optimizers.momentum_sgd.MomentumSGDRule(
                lr=optimizer.lr * 2, momentum=0)
    model.upscore2.disable_update()
    model.upscore8.disable_update()
    model.upscore_pool4.disable_update()

    # training loop

    trainer = fcn.Trainer(
        device=gpu,
        model=model,
        optimizer=optimizer,
        iter_train=iter_train,
        iter_valid=iter_valid,
        out=out,
        max_iter=100000,
    )
    trainer.train()


if __name__ == '__main__':
    main()