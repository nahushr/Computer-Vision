#!/usr/bin/env python3
from PIL import Image, ImageOps, ImageDraw
import numpy as np
from scipy import ndimage

##normalization function
def normalize(input_im):
    base = input_im.min()
    roof = input_im.max()
    diff = roof-base
    scale=diff /255
    input_im = input_im - base
    output = input_im/ scale
    return np . uint8 ( output )

##open image and convert to greyscale
image=Image.open("images/chess1.JPG")
temp=Image.open("images/chess1.JPG")
draw=ImageDraw.Draw(temp)
image=ImageOps.grayscale(image)

##creating the x and y kernels
kernel_x = np.array([[-1,1]])
kernel_y = np.array([[-1],[1]])

##image convolutions to compute the gradients along x and y directions
Ix=ndimage.convolve(image,kernel_x)
Iy=ndimage.convolve(image,kernel_y)

##create the Ix2,iy2,ix_iy
Ix=np.multiply(np.array(Ix),np.array(Ix))
Iy=np.multiply(np.array(Iy),np.array(Iy))
Ix_Iy=np.multiply(np.array(Ix),np.array(Iy))

##create the Aw,bw,cw by convolving with box filter
box_filter=np.matrix('1 1 1; 1 1 1; 1 1 1')
Aw=ndimage.convolve(Ix,box_filter)
Bw=ndimage.convolve(Ix_Iy,box_filter)
Cw=ndimage.convolve(Iy,box_filter)

##drawing the crosses on the image by computing the minimum eigenvalues as given in the hints of the pdf
for x in range(np.array(image).shape[1]):
    for y in range(np.array(image).shape[0]):
        e_values, e_vectors=np.linalg.eig(np.array([[Aw[y][x],Bw[y][x]],
                                             [Bw[y][x],Cw[y][x]]]))
        if(min(e_values)>120):
            draw.line(((x-5,y),(x+5,y)), fill=(255,0,0))
            draw.line(((x,y-5),(x,y+5)), fill=(255,0,0))

temp.save("output/Final_output.jpg")
out=Image.fromarray(normalize(Ix))
out.save("output/Ix2.jpg")
out=Image.fromarray(normalize(Iy))
out.save("output/Iy2.jpg")
out=Image.fromarray(Aw)
out.save("output/Aw.jpg")
out=Image.fromarray(Bw)
out.save("output/Bw.jpg")
out=Image.fromarray(Cw)
out.save("output/Cw.jpg")

