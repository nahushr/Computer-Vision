import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import sys
import json


def thread_method(cluster,):
    time_check = 0
    temp_cluster=cluster
    for key, value in cluster.items():
        print(time_check)
        time_check += 1
        # if(time_check>limit):
        #     break;
        pointa=np.array((key[0],key[1]))
        for key2, value2 in temp_cluster.items():
            # if (key2[0] - key[0] > 35 and key2[1] - key[1] > 35): break
            pointb=np.array((key2[0],key2[1]))
            distance=np.linalg.norm(pointa-pointb)
            if(distance<=35):
                cluster[(key[0], key[1])] = cluster[(key[0], key[1])] + 1  ## here we are counting how many points lies within a radius of 35 for that particular point
        cluster[(key[0], key[1])] = cluster[(key[0], key[1])] - 1  ## remove the same element added




def traces(im, image, i, j):
    ##go up
    up=0
    down = 0
    left=0
    right = 0
    for ii in range(i, image.shape[1]):
        for jj in range(j, j + 40):
            # (ii,jj)->staring point
            ##solve this by adding padding to image of whites
            try:
                if (im.getpixel((ii, jj)) < 120):
                    down = down + 1
            except:
                pass
        break;

    for ii in range(i, image.shape[1]):
        for jj in range(j-40, j):
            # (ii,jj)->staring point
            ##solve this by adding padding to image of whites
            try:
                if (im.getpixel((ii, jj)) < 120):
                    up = up + 1
            except:
                pass
        break;

    for jj in range(j, image.shape[0]):
        for ii in range(i, i + 40):
            try:
                if (im.getpixel((ii, jj)) < 120):
                    right = right + 1
            except:
                pass
        break;

    for jj in range(j, image.shape[0]):
        for ii in range(i-40, i):
            try:
                if (im.getpixel((ii, jj)) < 120):
                    left = left + 1
            except:
                pass
        break;

    # print(down,right)

    if (down > 8 and right > 8 and up>8 and left >8):
        return True
    else:
        return False


##driver code
im=Image.open("a-48.jpg")
# img = im.resize((2200, 1700), Image.ANTIALIAS)
image=np.array(im)
cluster={}
for i in range(image.shape[1]):
    for j in range(image.shape[0]):
        if(im.getpixel((i,j))<40): ##its a black pixel stop and traverse in all 4 directions
            if(traces(im,image,i,j)==True):
                cluster[(i,j)]=0

print("Total No. of coordinates: ", len(cluster))

img2=Image.new('RGB', (image.shape[1],image.shape[0]), color='white')
image2=np.array(img2)

# print(image2.shape)
# print(image.shape)
for i in range(image2.shape[1]):
    for j in range(image2.shape[0]):
        img2.putpixel((i,j),(im.getpixel((i,j)),im.getpixel((i,j)),im.getpixel((i,j)))) ## converting gray scale to 3 channnels of rgb so that we can put red pixel

for key,value in cluster.items():
    img2.putpixel((key[0],key[1]),(139,0,0))## putting red pixels

img2.save('red.jpg')
temp_cluster=cluster
## now we have all the coordinates of the red points lets cluster them together and find out how many points each has within a circle of radius 35
# print(cluster)
time_check=0
for key,value in cluster.items():
    print (time_check)
    time_check+=1
    for i in range(key[0]-12,key[0]+12):
        for j in range(key[1]-12,key[1]+12):
            r,g,b=img2.getpixel((i, j))
            if(r==139):
                cluster[(key[0],key[1])]=cluster[(key[0],key[1])]+1## here we are counting how many points lies within a radius of 35 for that particular point
    cluster[(key[0], key[1])] = cluster[(key[0], key[1])] - 1 ## remove the same element added
'''
exDict = {'exDict': cluster}

with open('file.txt', 'w') as file:
     file.write(json.dumps(exDict))
'''

f = open("dict.txt","w")
f.write( str(cluster) )
f.close()


draw=ImageDraw.Draw(img2)
for key,value in cluster.items():
    if value>150:
        draw.ellipse((key[0]-35,key[1]-35,key[0]+35,key[0]+35), fill='blue', outline='blue')
img2.save('circle.jpg')


