# naive_stereo.py
# This program performs block-based matching to create a depth map
# given 2 stereo images.

#Import the Image class from PIL (Pillow)
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import math

###########
###########

MAX_DISPARITY = 50    ### PLAY AROUND WITH THIS
WINDOW_SIZE = 13      ### PLAY AROUND WITH THIS


#Load left and right images
imL = Image.open("im0.png")  ### MODIFY THIS
imR = Image.open("im1.png")  ### MODIFY THIS

###########
###########

#Convert image modes
imL = ImageOps.grayscale(imL)
imR = ImageOps.grayscale(imR)

#Convert to numpy arrays
imL = np.array(imL,dtype='int64')
imR = np.array(imR,dtype='int64')


#possible disparity values
possible_disparities = range(0,MAX_DISPARITY)

#window size
safe_zone = int(WINDOW_SIZE/2)

disparity_matrix = np.zeros(imL.shape,dtype='int64')


def E_0(col,row,d):
    return np.sum((imL[row - safe_zone : row + safe_zone + 1, col-safe_zone + d: col + safe_zone + 1 + d]
                 - imR[row - safe_zone : row + safe_zone + 1, col-safe_zone : col + safe_zone + 1])**2)


def E(disparity_matrix):
    costs = np.full(imL.shape,np.inf)
    for d in possible_disparities:
        print("Calculating disparity",d, "/",possible_disparities[-1])

        for row in range(safe_zone, imL.shape[0] - safe_zone):
            for col in range(safe_zone, imL.shape[1] - safe_zone - d):
                local_result = E_0(col,row,d)

                if local_result < costs[row][col]:
                    costs[row][col] = local_result
                    disparity_matrix[row][col] = d
    return disparity_matrix


def normalize(input_im):
    base = input_im.min()
    roof = input_im.max()
    diff = roof - base
    scale = diff/255

    input_im = input_im - base
    output = input_im/scale

    return np.uint8(output)


disparity_matrix = normalize(E(disparity_matrix))


# save depth map as image (so that you can reuse it later!!!)
imDepth_image = Image.fromarray(disparity_matrix)
imDepth_image = imDepth_image.convert('RGB')
imDepth_image.save("depth.png")

def convolution(imR,image, weak_filter, medium_filter, strong_filter):
    image_padded=imR
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            if(image[j][i]>=101 and image[j][i]<=150):
                try:imR[j][i]=(weak_filter * image_padded[j:j + len(weak_filter), i:i + len(weak_filter)]).sum()
                except:pass
            elif(image[j][i]>=51 and image[j][i]<=100):
                try:imR[j][i]=(medium_filter * image_padded[j:j + len(medium_filter), i:i + len(medium_filter)]).sum()
                except:pass
            elif(image[j][i]>=0 and image[j][i]<=50):
                try:imR[j][i]=(strong_filter * image_padded[j:j + len(strong_filter), i:i + len(strong_filter)]).sum()
                except:pass
            image_padded = imR
    return Image.fromarray(imR)

image=Image.open("depth.png")
image=ImageOps.grayscale(image)
image=np.array(image)
weak_filter=(1/9)*np.ones((3,3))
medium_filter=(1/49)*np.ones((7,7))
strong_filter=(1/169)*np.ones((13,13))


convolution(np.array(imR),image,weak_filter,medium_filter,strong_filter).save("koech_effect_r.jpg")
convolution(np.array(imL),image,weak_filter,medium_filter,strong_filter).save("koech_effect_l.jpg")


