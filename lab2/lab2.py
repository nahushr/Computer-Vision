#!/usr/bin/env python3
from PIL import Image, ImageOps, ImageDraw
import numpy as np
from scipy import fftpack,ndimage,signal
import imageio
##step 1-> open greyscale and resize to same dimensions
image1,image2=Image.open("cat.jpg"),Image.open("man.jpg")
image1,image2=ImageOps.grayscale(image1),ImageOps.grayscale(image2)
image2=image2.resize((np.array(image1).shape[1],np.array(image1).shape[0]), Image.ANTIALIAS) ##resizing image2 to image1 size
##step2-> perform FFT
fft2_image1,fft2_image2=fftpack.fftshift(fftpack.fft2(image1)),fftpack.fftshift(fftpack.fft2(image2))
imageio.imsave('output/fft_image_1_cat.png', (numpy.log(abs(fft2_image1))* 255 /numpy.amax(numpy.log(abs(fft2_image1)))).astype(numpy.uint8))
imageio.imsave('output/fft_image_2_man.png', (numpy.log(abs(fft2_image2))* 255 /numpy.amax(numpy.log(abs(fft2_image2)))).astype(numpy.uint8))

##step3->
#assuming right now that the image is noise free so not removing any noise
#low pass filter on image1
pointb = np.array((int(np.array(fft2_image1).shape[0] / 2), int(np.array(fft2_image1).shape[1] / 2))) ##taking the image center
for i in range(np.array(fft2_image1).shape[0]):
    for j in range(np.array(fft2_image1).shape[1]):
        pointa=np.array((i,j))
        distance=np.linalg.norm(pointa-pointb)
        if(distance<=50):pass #keep that point
        else:fft2_image1[i][j]=0 # turn black
#high pass filter on image2
for i in range(np.array(fft2_image2).shape[0]):
    for j in range(np.array(fft2_image2).shape[1]):
        pointa,distance=np.array((i,j)),np.linalg.norm(pointa-pointb)
        if(distance>=50): pass#keep that point
        else: fft2_image2[i][j]=0# turn black
high_pass_image_result=abs(fftpack.ifft2(fftpack.ifftshift(fft2_image1)))
low_pass_image_result=abs(fftpack.ifft2(fftpack.ifftshift(fft2_image2)))
#pixel wise summation
new_image=np.add(np.array(high_pass_image_result),np.array(low_pass_image_result))
output=Image.fromarray(new_image)
if output.mode != 'RGB':
    output = output.convert('RGB')
output.save('output/Final_output_1.png')

## for the second image
pointb = np.array((int(np.array(fft2_image1).shape[0] / 2), int(np.array(fft2_image1).shape[1] / 2))) ##taking the image center
for i in range(np.array(fft2_image1).shape[0]):
    for j in range(np.array(fft2_image1).shape[1]):
        pointa=np.array((i,j))
        distance=np.linalg.norm(pointa-pointb)
        if(distance>=50):pass #keep that point
        else:fft2_image1[i][j]=0 # turn black
#high pass filter on image2
for i in range(np.array(fft2_image2).shape[0]):
    for j in range(np.array(fft2_image2).shape[1]):
        pointa,distance=np.array((i,j)),np.linalg.norm(pointa-pointb)
        if(distance<=50): pass#keep that point
        else: fft2_image2[i][j]=0# turn black
high_pass_image_result=abs(fftpack.ifft2(fftpack.ifftshift(fft2_image1)))
low_pass_image_result=abs(fftpack.ifft2(fftpack.ifftshift(fft2_image2)))
#pixel wise summation
new_image=np.add(np.array(high_pass_image_result),np.array(low_pass_image_result))
output=Image.fromarray(new_image)
if output.mode != 'RGB':
    output = output.convert('RGB')
output.save('output/Final_output_2.png')



