# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
from datasets.mask_util import draw_ann
import util
def get_minibatch(roidb, num_classes):
  
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  im_blob, mask_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

  
  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"
  
  # gt boxes: (x1, y1, x2, y2, cls)
  if cfg.TRAIN.USE_ALL_GT:
    # Include all ground truth boxes
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
    gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
  gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
  mask_blob = mask_blob[:, :, :, gt_inds]  
  
  for bi in len(gt_boxes):
    box_cls = gt_boxes[:, 4]
    m = mask_blob[:, :, :, bi];
    check_set = set(np.flatten(m));
    np.testing.assert_(len(check_set) <= 2, check_set);
    np.testing.assert_(box_cls in check_set, (box_cls, check_set));
    mask_blob[:, :, :, bi] = (m == box_cls) * 1
  
  blobs = {'data': im_blob, 'mask': mask_blob}
  blobs['gt_boxes'] = gt_boxes
  blobs['im_info'] = np.array(
    [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
    dtype=np.float32)
  return blobs

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  processed_ims = []
  processed_masks = [];
  im_scales = []
  
  #only coco has segs.
  with_mask = 'segs' in roidb[0];
  
  segs = roidb[0]['segs'];
  """
  for si1, _seg in enumerate(segs):
    if not type(_seg) == list:
      util.io.dump('~/temp_nfs/no-use/data_seg_dict.pkl', [roidb, scale_inds])
      import pdb
      pdb.set_trace()
    else:
      util.io.dump('~/temp_nfs/no-use/data_seg_list.pkl', [roidb, scale_inds])
  """
  for i in range(num_images):
    im = cv2.imread(roidb[i]['image'])
    ori_height, ori_width = im.shape[0:2];
    
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
    if with_mask:                    
      new_height, new_width = im.shape[0:2];
      mask = draw_ann(roidb[i], ori_height, ori_width, new_height, new_width);

    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
      if with_mask:
        mask = mask[:, ::-1, :]


    im_scales.append(im_scale)
    processed_ims.append(im)
    if with_mask:
      processed_masks.append(mask);
    
  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims);
  mask_blob = [];
  if with_mask:
    mask_blob = im_list_to_blob(processed_masks);
  return blob, mask_blob, im_scales
