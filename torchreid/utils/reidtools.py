from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import os.path as osp
import shutil
from PIL import Image,ImageDraw
from .iotools import mkdir_if_missing


def visualize_ranked_results(label, distmat, dataset, save_dir='log/ranked_results', topk=20):
    """
    Visualize ranked results

    Support both imgreid and vidreid

    Args:
    - distmat: distance matrix of shape (num_query, num_gallery).
    - dataset: a 2-tuple containing (query, gallery), each contains a list of (img_path, pid, camid);
               for imgreid, img_path is a string, while for vidreid, img_path is a tuple containing
               a sequence of strings.
    - save_dir: directory to save output images.
    - topk: int, denoting top-k images in the rank list to be visualized.
    """
    num_q, num_g = distmat.shape

    print("Visualizing top-{} ranks".format(topk))
    print("# query: {}\n# gallery {}".format(num_q, num_g))
    print("Saving images to '{}'".format(save_dir))
    
    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)
    
    indices = np.argsort(distmat, axis=1)
    mkdir_if_missing(save_dir)

    def _cp_img_to(src, dst, rank, prefix):
        """
        - src: image path or tuple (for vidreid)
        - dst: target directory
        - rank: int, denoting ranked position, starting from 1
        - prefix: string
        """
        if isinstance(src, tuple) or isinstance(src, list):
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
            shutil.copy(src, dst)

    for q_idx in range(num_q):
        # data, label, id, camid, name, sem
        _, qlabel, qpid, qcamid, qimg_path, qsem = query[q_idx]
        qlabel = ''.join(map(str, qlabel))
        qdir = osp.join(save_dir, 'sem_'+str(qsem)+'_'+qlabel)
        mkdir_if_missing(qdir)
        img = Image.new('RGB', (130, 400), color='black')
        d = ImageDraw.Draw(img)
        test_label = label
        d.text((10, 10), 'Query Attributes', fill=(255, 255, 0))
        y = 30
        for i in range(len(test_label)):
            d.text((10, y), test_label[i].ljust(15) + ':  '+qlabel[i], fill=(255, 255, 0))
            y += 12
        qimgdir = osp.join(qdir, 'query_top000_'+str(qsem)+'_exp_'+str(qpid)+'.png')
        img.save(qimgdir)

        rank_idx = 1
        for g_idx in indices[q_idx,:]:
            _, glabel, gpid, gcamid, gimg_path, gsem = gallery[g_idx]
            _cp_img_to(gimg_path, qdir, rank=rank_idx, prefix='gallery_'+str(gsem))
            rank_idx += 1
            if rank_idx > topk:
                break

    print("Done")


def visualize_ranked_results_train(label, distmat, dataset, save_dir='log/ranked_results', topk=20):
    """
    Visualize ranked results

    Support both imgreid and vidreid

    Args:
    - distmat: distance matrix of shape (num_query, num_gallery).
    - dataset: a 2-tuple containing (query, gallery), each contains a list of (img_path, pid, camid);
               for imgreid, img_path is a string, while for vidreid, img_path is a tuple containing
               a sequence of strings.
    - save_dir: directory to save output images.
    - topk: int, denoting top-k images in the rank list to be visualized.
    """
    num_q, num_g = distmat.shape

    print("Visualizing top-{} ranks".format(topk))
    print("# query: {}\n# gallery {}".format(num_q, num_g))
    print("Saving images to '{}'".format(save_dir))

    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)

    indices = np.argsort(distmat, axis=1)
    mkdir_if_missing(save_dir)

    def _cp_img_to(src, dst, rank, prefix):
        """
        - src: image path or tuple (for vidreid)
        - dst: target directory
        - rank: int, denoting ranked position, starting from 1
        - prefix: string
        """
        if isinstance(src, tuple) or isinstance(src, list):
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
            shutil.copy(src, dst)

    for q_idx in range(num_q):
        # data, i, label, id, cam, name, sem
        _, _, qlabel, qpid, qcamid, qimg_path, qsem = query[q_idx]
        qlabel = ''.join(map(str, qlabel))
        qdir = osp.join(save_dir, 'sem_' + str(qsem) + '_' + qlabel)
        mkdir_if_missing(qdir)
        img = Image.new('RGB', (130, 400), color='black')
        d = ImageDraw.Draw(img)
        test_label = label
        d.text((10, 10), 'Query Attributes', fill=(255, 255, 0))
        y = 30
        for i in range(len(test_label)):
            d.text((10, y), test_label[i].ljust(15) + ':  ' + qlabel[i], fill=(255, 255, 0))
            y += 12
        qimgdir = osp.join(qdir, 'query_top000_' + str(qsem) + '_exp_' + str(qpid) + '.png')
        img.save(qimgdir)

        rank_idx = 1
        for g_idx in indices[q_idx, :]:
            # data, label, id, camid, img_path, sem
            _, glabel, gpid, gcamid, gimg_path, gsem = gallery[g_idx]
            _cp_img_to(gimg_path, qdir, rank=rank_idx, prefix='gallery_' + str(gsem))
            rank_idx += 1
            if rank_idx > topk:
                break

    print("Done")
