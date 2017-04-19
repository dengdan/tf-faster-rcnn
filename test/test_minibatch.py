import util
import numpy as np
util.mod.add_to_path('../lib');
util.mod.add_to_path('../data/coco/PythonAPI')
import roi_data_layer.minibatch as minibatch
def test_get_image_blob():
  roidb, scale_inds = util.io.load('~/temp_nfs/no-use/data_seg_dict.pkl')
#  roidb, scale_inds = util.io.load('~/temp_nfs/no-use/data_seg_list.pkl')
  print roidb[0]['image']
#  import pdb
#  pdb.set_trace()
  
  blob, mask_blob, im_scales = minibatch._get_image_blob(roidb, scale_inds);
  image = np.asarray(blob[0, ...], dtype = np.int32);
  mask = np.asarray(mask_blob[0, ...], dtype = np.int32);
  util.plt.show_images(images = [image, mask], titles = ["Image", "Mask"], save = True, show = False, path = '~/temp_nfs/no-use/test.jpg');

if __name__ == "__main__":
  test_get_image_blob();
