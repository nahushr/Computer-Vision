# To split the given images in dataset into respective Dya & Night Instances.
import numpy as np

from PIL import Image
import glob


BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


counter = 1

for filename in glob.glob("training/*.jpg"):
  im = Image.open(filename)
  im_arr = np.array(im)
  width,height = im.size
  print(width,height)
  width = width//2

  a = im_arr[:, width:]
  b = im_arr[:, :width]
  im1 = Image.fromarray(a)
  im2 = Image.fromarray(b)

  file_str = str(counter)
  file1_path = "day/"+file_str+".png"
  file2_path = "night/"+file_str+".png"
  im1.save(file1_path, 'PNG')
  im2.save(file2_path, 'PNG')

  counter+=1