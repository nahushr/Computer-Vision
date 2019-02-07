import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import sys
from collections import OrderedDict
from matplotlib.patches import Circle


def traces(im, image, i, j):
    ##go up
    up=0
    down = 0
    left=0
    right = 0
    for ii in range(i, image.shape[1]):
        for jj in range(j, j + 60):
            # (ii,jj)->staring point
            ##solve this by adding padding to image of whites
            try:
                if (im.getpixel((ii, jj)) < 120):
                    down = down + 1
            except:
                pass
        break;

    for ii in range(i, image.shape[1]):
        for jj in range(j-60, j):
            # (ii,jj)->staring point
            ##solve this by adding padding to image of whites
            try:
                if (im.getpixel((ii, jj)) < 120):
                    up = up + 1
            except:
                pass
        break;

    for jj in range(j, image.shape[0]):
        for ii in range(i, i + 60):
            try:
                if (im.getpixel((ii, jj)) < 120):
                    right = right + 1
            except:
                pass
        break;

    for jj in range(j, image.shape[0]):
        for ii in range(i-60, i):
            try:
                if (im.getpixel((ii, jj)) < 120):
                    left = left + 1
            except:
                pass
        break;

    # print(down,right)

    if (down > 10 and right > 10 and up>10 and left >10):
        return True
    else:
        return False


##driver code
im=Image.open("a-48.jpg")
# img = im.resize((2200, 1700), Image.ANTIALIAS)
image=np.array(im)
row_list=[]
col_list=[]
answer_list_x=[]
answer_list_y=[]

final_answer_list_x=[]
final_answer_list_y=[]

dict_row={}
dict_col={}

up,down,left,right=[],[],[],[]
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
time_check=0
for key,value in cluster.items():
    print(time_check)
    time_check+=1
    # pointa=np.array((key[0],key[1]))
    for key2,value2 in temp_cluster.items():
        if(key2[0]-key[0]>35 and key2[1]-key[1]>35):break
        # pointb=np.array((key2[0],key2[1]))
        # distance=np.linalg.norm(pointa-pointb)
        # if(distance<=35):
        cluster[(key[0],key[1])]=cluster[(key[0],key[1])]+1## here we are counting how many points lies within a radius of 35 for that particular point
    cluster[(key[0], key[1])] = cluster[(key[0], key[1])] - 1 ## remove the same element added
cluster_final = sorted(cluster, key=cluster.get, reverse=True)

print(cluster_final)

##this part is on;y for visual representation
draw=ImageDraw.Draw(img2)
for key,value in cluster_final.items():
    if value>150:
        draw.ellipse((key[0]-35,key[1]-35,key[0]+35,key[0]+35), fill='blue', outline='blue')
img2.save('circle.jpg')

