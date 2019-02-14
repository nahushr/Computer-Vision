#!/usr/bin/env python3
from PIL import Image, ImageOps
import numpy as np
from scipy import fftpack
import warnings
warnings.filterwarnings('ignore') ##nothing but to supress warning of complex numbers
##this program takes around 20 seconds to run
def boxing(): ##function to get filters and in order to create the final output
    fft2_image1, fft2_image2 = fftpack.fftshift(fftpack.fft2(image1)), fftpack.fftshift(fftpack.fft2(image2))
    pointb = np.array((int(np.array(fft2_image1).shape[0] / 2), int(np.array(fft2_image1).shape[1] / 2)))  ##taking the image center
    #low pass filter
    for i in range(np.array(fft2_image1).shape[0]):
        for j in range(np.array(fft2_image1).shape[1]):
            pointa = np.array((i, j))
            distance = np.linalg.norm(pointa - pointb)
            if (distance <= 30):pass
                # temp1[i][j]=fft2_image1[i][j]
            else:
                fft2_image1[i][j] = 0
    # high pass filter
    for i in range(np.array(fft2_image2).shape[0]):
        for j in range(np.array(fft2_image2).shape[1]):
            pointa, distance = np.array((i, j)), np.linalg.norm(pointa - pointb)
            if (distance >= 30):pass
                  # temp2[i][j]=fft2_image1[i][j]
            else:
                fft2_image2[i][j]=0
    return fft2_image1, fft2_image2

def boxing2():## function to reverse the filters roles on the images in order to create the final output
     fft2_image1, fft2_image2 = fftpack.fftshift(fftpack.fft2(image1)), fftpack.fftshift(fftpack.fft2(image2))
     pointb = np.array(
         (int(np.array(fft2_image1).shape[0] / 2), int(np.array(fft2_image1).shape[1] / 2)))  ##taking the image center
     # high pass filter
     for i in range(np.array(fft2_image1).shape[0]):
         for j in range(np.array(fft2_image1).shape[1]):
             pointa = np.array((i, j))
             distance = np.linalg.norm(pointa - pointb)
             if (distance >= 30):
                 pass
             # temp1[i][j]=fft2_image1[i][j]
             else:
                 fft2_image1[i][j] = 0
     # low pass filter
     for i in range(np.array(fft2_image2).shape[0]):
         for j in range(np.array(fft2_image2).shape[1]):
             pointa, distance = np.array((i, j)), np.linalg.norm(pointa - pointb)
             if (distance <= 30):
                 pass
             # temp2[i][j]=fft2_image1[i][j]
             else:
                 fft2_image2[i][j] = 0
     return fft2_image1,fft2_image2

##step 1-> open greyscale and resize to same dimensions
image1,image2=Image.open("cat.jpg"),Image.open("man.jpg")
image1,image2=ImageOps.grayscale(image1),ImageOps.grayscale(image2)
image2=image2.resize((np.array(image1).shape[1],np.array(image1).shape[0]), Image.ANTIALIAS) ##resizing image2 to image1 size

##step2-> perform FFT
fft2_image1, fft2_image2 = fftpack.fftshift(fftpack.fft2(image1)), fftpack.fftshift(fftpack.fft2(image2))
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
t1,t2=boxing()
t3,t4=boxing2()
high_pass_image_result1=abs(fftpack.ifft2(fftpack.ifftshift(t1)))
low_pass_image_result1=abs(fftpack.ifft2(fftpack.ifftshift(t2)))

high_pass_image_result2=abs(fftpack.ifft2(fftpack.ifftshift(t3)))
low_pass_image_result2=abs(fftpack.ifft2(fftpack.ifftshift(t4)))

#step4->
#pixel wise summation and saving the final output
new_image1=np.add(np.array(high_pass_image_result1),np.array(low_pass_image_result1))
output1=Image.fromarray(new_image1)
if output1.mode != 'RGB':
    output1 = output1.convert('RGB')
output1.save("output_Final_output_1.png")

new_image2=np.add(np.array(high_pass_image_result2),np.array(low_pass_image_result2))
output2=Image.fromarray(new_image2)
if output2.mode != 'RGB':
    output2 = output2.convert('RGB')
output2.save("output_Final_output_2.png")
