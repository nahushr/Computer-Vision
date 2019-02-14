#!/usr/bin/env python3
##the working of my code is explained below step by stepin comments
##the secret message was HI and an inverted HI
from scipy import fftpack
import imageio
import numpy
import warnings
warnings.filterwarnings('ignore')## basically to ignore runtime warning of converting complex to integer to find the pixel position in the fourier domain

# load in an image, convert to grayscale if needed
image = imageio.imread('pichu-secret-1.png',as_gray=True)

# take the fourier transform of the image
fft2 = fftpack.fftshift(fftpack.fft2(image))

##first we initialize two starting coordinates
x1,y1=0,0
x2,y2=0,0

##traversing the fft array pixel by pixel we pin point the last point in the space which has a frequency of 18000+ which lies in the unusual range
##we store these coordinated in x1 and y1
for i in range(fft2.shape[1]):
    for j in range(fft2.shape[0]):
        if int(fft2[i][j])>18000:
            x1=i
            y1=j
            break;

##traversing the fft array pixel by pixel to find the first coordinates of piexels with a frequency of 18000+ which is a general threshold set by me after various testings with different values
##the coordinates are then pin pointed to x2,y2
control_the_loop=False
for i in range(fft2.shape[0]):
    for j in range(fft2.shape[1]):
        if(int(fft2[i][j])>18000 and control_the_loop==False):
            x2=i
            y2=j
            control_the_loop=True
            break
    if(control_the_loop==True):
        break

##now since we have the coordinates x1,y1 and x2,y2 we draw a square covering 12 pixels in length starting from x1,y1 to the bottom right
##and starting from x2,y2 to the top left
##these squares built cover the secret message and thus it removes the noise from the final fft-then-ifft.png image
for i in range(x1-12,x1):
    for j in range(y1,y1+12):
        fft2[i][j]=0

for i in range(x2,x2+12):
    for j in range(y2,y2+12):
        fft2[i][j]=0



# save FFT to a file. To help with visualization, we take
# the log of the magnitudes, and then stretch them so they
# fill the whole range of pixel values from 0 to 255.
imageio.imsave('output/fft.png', (numpy.log(abs(fft2))* 255 /numpy.amax(numpy.log(abs(fft2)))).astype(numpy.uint8))

# At this point, fft2 is just a numpy array and you can
# modify it in order to modify the image in the frequency
# space. Here's a little example (that makes a nearly
# imperceptible change, but demonstrates what you can do.

# fft2[1,1]=fft2[1,1]+1

# now take the inverse transform to convert back to an image
ifft2 = abs(fftpack.ifft2(fftpack.ifftshift(fft2)))

# and save the image
imageio.imsave('output/fft-then-ifft.png', ifft2.astype(numpy.uint8))

