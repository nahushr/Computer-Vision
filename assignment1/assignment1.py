##code solely created by Nahush Raichura
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import sys
import json

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
print("Processing the image and plotting the red pixels..........")
im=Image.open("c-18.jpg")
# img = im.resize((2200, 1700), Image.ANTIALIAS)
image=np.array(im)
cluster={}
for i in range(image.shape[1]):
    for j in range(image.shape[0]):
        if(im.getpixel((i,j))<40): ##its a black pixel stop and traverse in all 4 directions
            if(traces(im,image,i,j)==True):
                cluster[(i,j)]=0

print("Total No. of red coordinates: ", len(cluster))
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
print("Clustering the red points to construct blue circles..........")
temp_cluster=cluster
## now we have all the coordinates of the red points lets cluster them together and find out how many points each has within a circle of radius 35
# print(cluster)
#time_check=0
for key,value in cluster.items():
    #print (time_check)
    #time_check+=1
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

threshold=int(len(cluster)/480)
draw=ImageDraw.Draw(img2)
for key,value in cluster.items():
    if value>threshold:
        draw.ellipse((key[0]-12,key[1]-12,key[0]+12,key[1]+12), fill='blue')
img2.save('circle.jpg')
print("Creating a grid system......")

column={}
row={}
for i in range(np.array(img2).shape[0]):
    for j in range(np.array(img2).shape[1]):
        r,g,b=img2.getpixel((j, i))
        if (r==0 and g==0 and b==255):
            # print("hello")
            if i in row:
                row[i]+=1
            else:
                row[i]=0

for i in range(np.array(img2).shape[1]):
    for j in range(np.array(img2).shape[0]):
        r,g,b=img2.getpixel((i, j))
        if (r==0 and g==0 and b==255 and j>400):
            # print("hello")
            if i in column:
                column[i]+=1
            else:
                column[i]=0
#print("Total blue column dictionary: ",column)
print("Total blue column dictionary length: ",len(column))
#print("Total blue row dictionary: ",row)
print("Total blue row dictionary length: ",len(row))

y_list=[]
x_list=[]
#for columns
previous=0
for key, value in column.items():
    draw=ImageDraw.Draw(img2)
    if(int(key)-int(previous)>45 and value>200):
        draw.line((key,0,key,np.array(img2).shape[0]), fill=128)
        previous=key
        y_list.append(key)

keys=list(row.keys())

for i in range(len(keys)):
    if int(keys[i])>500:
        start=int(keys[i])
        break

threshold=(int(keys[len(keys)-1])-int(start))/29

#for rows
previous=0
for key, value in row.items():
    draw=ImageDraw.Draw(img2)
    if(int(key)-int(previous)>int(threshold) and value>50 and key>500):
        draw.line((0,key,np.array(img2).shape[1],key), fill=128)
        previous=key
        x_list.append(key)
        
print("Intersection y list: ",len(y_list))
print("Intersection x list: ",len(x_list))

img2.save('grid.jpg')
print("Calculating the answers from the grid system.........")
answer_list={}
for i in range(1,88):
    answer_list[str(i)]=[]

for i in range(len(x_list)):
    for j in range(len(y_list)):
        counter=0
        for x in range(x_list[i],x_list[i]+30):
            for y in range(y_list[j],y_list[j]+30):
                r,g,b=img2.getpixel((y,x))
                if(r==0 and g==0 and b==255):
                    ## get the blue pixel if there are more than 20 pixels then its the answer
                    counter+=1
                    if(counter>=20):
                        if(j>=0 and j<=4):
                            key=str(i+1)
                            if(j==0):answer_list[key].append('A')
                            elif(j==1):answer_list[key].append('B')
                            elif(j==2):answer_list[key].append('C')
                            elif(j==3):answer_list[key].append('D')
                            elif(j==4):answer_list[key].append('E')

                        elif(j>=5 and j<=9):
                            key=(str(i+1+29))
                            if (j == 5):
                                answer_list[key].append('A')
                            elif (j == 6):
                                answer_list[key].append('B')
                            elif (j == 7):
                                answer_list[key].append('C')
                            elif (j == 8):
                                answer_list[key].append('D')
                            elif (j == 9):
                                answer_list[key].append('E')
                        elif(j>=10 and j<=14):
                            key=(str(i+1+29+29))
                            answer_list[key]
                            if (j == 10):
                                answer_list[key].append('A')
                            elif (j == 11):
                                answer_list[key].append('B')
                            elif (j == 12):
                                answer_list[key].append('C')
                            elif (j == 13):
                                answer_list[key].append('D')
                            elif (j == 14):
                                answer_list[key].append('E')

for key,value in answer_list.items():
    mylist=list(dict.fromkeys(value))
    answer_list[key]=mylist

print("Final answer dictionary.........")
print(answer_list) 

sys.exit(1)

