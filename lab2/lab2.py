#!/usr/bin/env python3
from PIL import Image, ImageOps, ImageDraw
import numpy as np
from scipy import fftpack,ndimage,signal
##this program takes around 20 seconds to run

##step 1-> open greyscale and resize to same dimensions
image1,image2=Image.open("cat.jpg"),Image.open("man.jpg")
image1,image2=ImageOps.grayscale(image1),ImageOps.grayscale(image2)
image2=image2.resize((np.array(image1).shape[1],np.array(image1).shape[0]), Image.ANTIALIAS) ##resizing image2 to image1 size

##step2-> perform FFT
fft2_image1,fft2_image2=fftpack.fftshift(fftpack.fft2(image1)),fftpack.fftshift(fftpack.fft2(image2))
##saving fft image 1
fft_image1_output=(np.log(abs(fft2_image1))* 255 /np.amax(np.log(abs(fft2_image1)))).astype(np.uint8)
fft_image1_output=Image.fromarray(fft_image1_output)
fft_image1_output.save('output_fft_image_1_cat.png')
##saving fft image 2
fft_image2_output=(np.log(abs(fft2_image2))* 255 /np.amax(np.log(abs(fft2_image2)))).astype(np.uint8)
fft_image2_output=Image.fromarray(fft_image2_output)
fft_image2_output.save('output_fft_image_2_man.png')

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

##step4->
#pixel wise summation
new_image=np.add(np.array(high_pass_image_result),np.array(low_pass_image_result))
output=Image.fromarray(new_image)
if output.mode != 'RGB':
    output = output.convert('RGB')
output.save('output_Final_output_1.png')

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

##step4->
#pixel wise summation
new_image=np.add(np.array(high_pass_image_result),np.array(low_pass_image_result))
output=Image.fromarray(new_image)
if output.mode != 'RGB':
    output = output.convert('RGB')
output.save('output_Final_output_2.png')



