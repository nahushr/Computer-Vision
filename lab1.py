#!/usr/bin/env python3
from PIL import Image, ImageFilter
import numpy as np
import sys

def convolution(image, kernel_matrix, kernel_dimensions):
    output=np.zeros_like(image) ## creating a null output
    image_padded=np.zeros((image.shape[0]+kernel_dimensions-1, image.shape[1]+kernel_dimensions-1, 3)) ## I'm adding kernel dimension-1 padding of zeros to the image to handle border convolution ## this generalizes it as any kernel can be used then
    image_padded[1:-(kernel_dimensions-2), 1:-(kernel_dimensions-2)]=image ##getting rest of the data from the image
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            ## now I'm convolution over each red, green and blue pixel this automatically crops thepadded edges after computing it
            red=((kernel_matrix * image_padded[j:j+kernel_dimensions, i:i+kernel_dimensions, 0]).sum())
            green=((kernel_matrix * image_padded[j:j+kernel_dimensions, i:i+kernel_dimensions, 1]).sum())
            blue=((kernel_matrix * image_padded[j:j+kernel_dimensions, i:i+kernel_dimensions, 2]).sum())
            output[j,i,0]=red
            output[j,i,1]=blue
            output[j,i,2]=green
    return Image.fromarray(output)

##main driver code
im=Image.open("w.jpg")
kernel_matrix1=np.array([[0,0,0],[0,1,0],[0,0,0]])
kernel_matrix2=np.array([[0.111111,0.111111,0.111111],[0.111111,0.111111,0.111111],[0.111111,0.111111,0.111111]])
kernel_matrix3=np.array([[0.003,0.013,0.022,0.013,0.003],[0.013,0.059,0.097,0.059,0.013],[0.022,0.097,0.159,0.097,0.022],[0.013,0.059,0.097,0.059,0.013],[0.003,0.013,0.022,0.013,0.003]])
kernel_matrix4_temp=np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]])
alpha=0.5
temp_mat_value=1+alpha
kernel_matrix4=np.array(np.subtract(temp_mat_value*kernel_matrix4_temp, kernel_matrix3))

#orignal image
im.show()

#identity kernel
result=convolution(np.array(im),kernel_matrix1,3)
result.save('Identity_kernel.jpeg')
result.show()

#box blur kernel
result=convolution(np.array(im),kernel_matrix2,3)
result.save('box_blur_filter.jpeg')
result.show()

#gaussian kernel
result=convolution(np.array(im),kernel_matrix3,5)
result.save('approximated_gaussian_filter.jpeg')
result.show()

#sharpening kernel
result=convolution(np.array(im),kernel_matrix4,5)
result.save('sharpening_filter.jpeg')
result.show()

############using pillow filtering libraries:
##identity kernel
result=im.filter(ImageFilter.Kernel((3,3),[0,0,0,0,1,0,0,0,0]))
result.save('identity_kernel_direct.jpeg')
result.show()

##box blur filter filter
result=im.filter(ImageFilter.Kernel((3,3),[0.111111,0.111111,0.111111,0.111111,0.111111,0.111111,0.111111,0.111111,0.111111]))
result.save('box_blur_filter_direct.jpeg')
result.show()

##approximated gaussian filter
result=im.filter(ImageFilter.Kernel((5,5),[0.003,0.013,0.022,0.013,0.003,0.013,0.059,0.097,0.059,0.013,0.022,0.097,0.159,0.097,0.022,0.013,0.059,0.097,0.059,0.013,0.003,0.013,0.022,0.013,0.003]))
result.save('approximated_gaussian_filter_direct.jpeg')
result.show()

##sharpening filter
result=im.filter(ImageFilter.Kernel((5,5),[-0.003, -0.013, -0.022, -0.013, -0.003,-0.013, -0.059, -0.097, -0.059, -0.013,-0.022, -0.097, 1.341, -0.097, -0.022,-0.013, -0.059, -0.097, -0.059, -0.013,-0.003, -0.013, -0.022, -0.013, -0.003]))
result.save('sharpening_filter_direct.jpeg')
result.show()


