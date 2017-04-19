import numpy as np
import cv2

from pycocotools import mask as maskUtils
import util
def draw_mask(seg, height, width, t_height, t_width, color):
    if type(seg) == list:
        # polygon
        m = util.img.black((height, width))
        for _seg in seg:
            poly = np.array(_seg).reshape((int(len(_seg)/2), 2))
            cnt = util.img.points_to_contours(poly);
            util.img.draw_contours(m, cnt, idx = -1, color = 1, border_width = -1)
    else:
        # mask
        if type(seg['counts']) == list:
            rle = maskUtils.frPyObjects([seg], height, width)
        else:
            rle = [seg]
        m = maskUtils.decode(rle)
        m = m[..., 0]
    m = cv2.resize(m, dsize = (t_height, t_width), interpolation = cv2.INTER_NEAREST)
    np.testing.assert_(np.min(m) >= 0, ('np.min(m)', np.min(m)))  
    np.testing.assert_(np.max(m) == 1, ('np.max(m)', np.max(m)))
    m = m * color;
    return m;
    
def draw_ann_in_one(ann, height, width, t_height, t_width):
  masks = [];
  segs = ann['segs']
  gt_cls = ann['gt_classes']
  for si, seg in enumerate(segs):
    masks.append(draw_mask(seg, height, width, t_height, t_width, int(gt_cls[si])));
  
  ret = masks[0];
  for i in range(1, len(masks)):
    non_zeros1 = ret > 0;
    mask = masks[i];
    non_zeros2 = mask > 0;
    non_zeros = non_zeros1 + non_zeros2;
    overlapped = non_zeros == 2;
    non_zeros = non_zeros == 1;
    non_zeros1 = non_zeros1 * non_zeros;
    non_zeros2 = non_zeros2 * (non_zeros + overlapped);
    ret = ret * non_zeros1 + mask * non_zeros2
  return ret

def draw_ann(ann, height, width, t_height, t_width):
  masks = [];
  segs = ann['segs']
  gt_cls = ann['gt_classes']
  for si, seg in enumerate(segs):
    masks.append(draw_mask(seg, height, width, t_height, t_width, int(gt_cls[si])));
  
  return masks

