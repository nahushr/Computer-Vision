#!/usr/bin/env python3
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import sys
import json
## the traces function is used to find the traces of an image around the pixel
def traces(im, image, i, j):
    up,down,right,left = 0,0,0,0
    ##this for loop is used to traverse 40 pixels to down and get the total number of black(<120) pixels
    for ii in range(i, image.shape[1]):
        for jj in range(j, j + 40):
            # (ii,jj)->staring point
            try:
                if (im.getpixel((ii, jj)) < 200):down = down + 1
            except:pass
        break
    ##this for loop is used to traverse 40 pixels up to get the total number of(<120) black pixels
    for ii in range(i, image.shape[1]):
        for jj in range(j - 40, j):
            # (ii,jj)->staring point
            try:
                if (im.getpixel((ii, jj)) < 200):up = up + 1
            except:pass
        break
    ##this for loop is used to traverse 40 pixels to right and get the total number of black(<120) pixels
    for jj in range(j, image.shape[0]):
        for ii in range(i, i + 40):
            try:
                if (im.getpixel((ii, jj)) < 200):right = right + 1
            except:pass
        break
    ##this for loop is used to traverse 40 pixels to left and get the total number of black(<120) pixels
    for jj in range(j, image.shape[0]):
        for ii in range(i - 40, i):
            try:
                if (im.getpixel((ii, jj)) < 200):left = left + 1
            except:pass
        break
    ##if there are 8 black pixels in each direction traversed then we return true or else false
    if (down > 8 and right > 8 and up > 8 and left > 8):return True
    else:return False
##this function is used to add column lines to make a total of 15 lines on the image for creating the grid system in the image
def func(my_list, position, i, j):
    if (position == 'r'):my_list.insert(i + 1, my_list[i] + 55)
    if (position == 'l'):my_list.insert(j, my_list[j] - 60)
    return my_list
##driver code
if len(sys.argv) < 4:
    print("Usage: \n./grade.py form.jpg output.jpg output.txt")
    sys.exit()
# print("Processing the image and plotting the red pixels..........")
im = Image.open(sys.argv[1])
#im=im.filter(ImageFilter.SMOOTH)
#im=im.filter(ImageFilter.SHARPEN)

image = np.array(im)
cluster = {}
#Traversing the image array pixel by pixel to get the red points
for i in range(image.shape[1]):
    for j in range(image.shape[0]):
        if (im.getpixel((i, j)) < 40):  ##its a black pixel stop and traverse in all 4 directions
            if (traces(im, image, i, j) == True):cluster[(i, j)] = 0
# print("Total No. of red coordinates: ", len(cluster))
##this converts a greyscale image to a rgb channel so that we can put different colors on the image
img2 = Image.new('RGB', (image.shape[1], image.shape[0]), color='white')
image2 = np.array(img2)
#converting each pixel from greyscale to 3 rgb channel we put the same greyscale value to each r,g,b panel
for i in range(image2.shape[1]):
    for j in range(image2.shape[0]):img2.putpixel((i, j), (im.getpixel((i, j)), im.getpixel((i, j)), im.getpixel((i, j))))  ## converting gray scale to 3 channnels of rgb so that we can put red pixel
#we put red pixels from the cluster dictionary , this puts a lot of red points in the answers box
for key, value in cluster.items():img2.putpixel((key[0], key[1]), (139, 0, 0))  ## putting red pixels
img2.save('red.jpg')
# print("Clustering the red points to construct blue circles..........")
temp_cluster = cluster
## now we have all the coordinates of the red points lets cluster them together and find out how many points each has within a circle of radius 12
for key, value in cluster.items():
    for i in range(key[0] - 12, key[0] + 12):
        for j in range(key[1] - 12, key[1] + 12):
            r, g, b = img2.getpixel((i, j))
            if (r == 139):cluster[(key[0], key[1])] = cluster[(key[0], key[1])] + 1  ## here we are counting how many points lies within a radius of 35 for that particular point
    cluster[(key[0], key[1])] = cluster[(key[0], key[1])] - 1  ## remove the same element added
##drawing blue circle around the a square box of the points found in the cluster dictionary this removes the outliers of red and gets us blue circles only around the main concentrated region of red pixels so that we get blue circles only over the marked answers
threshold = int(len(cluster) / 480) ##putting a threshold
draw = ImageDraw.Draw(img2)
for key, value in cluster.items():
    if value > threshold:
        draw.ellipse((key[0] - 12, key[1] - 12, key[0] + 12, key[1] + 12), fill='blue')
img2.save('circle.jpg')
# print("Creating a grid system......")
##getting the row coordinates for drawing the horizontal lines
column,row = {},{}
for i in range(np.array(img2).shape[0]):
    for j in range(np.array(img2).shape[1]):
        r, g, b = img2.getpixel((j, i))
        if (r == 0 and g == 0 and b == 255):
            if i in row:row[i] += 1
            else:row[i] = 0
##getting the column coordinates for drawing the vertical lines 
for i in range(np.array(img2).shape[1]):
    for j in range(np.array(img2).shape[0]):
        r, g, b = img2.getpixel((i, j))
        if (r == 0 and g == 0 and b == 255 and j > 400):
            if i in column:column[i] += 1
            else:column[i] = 0
# print("Total blue column dictionary: ",column)
# print("Total blue column dictionary length: ", len(column))
# print("Total blue row dictionary: ",row)
# print("Total blue row dictionary length: ", len(row))
y_list,x_list = [],[]
# for columns we draw the vertical lines
previous = 0
for key, value in column.items():
    draw = ImageDraw.Draw(img2)
    if (int(key) - int(previous) > 60 and value > 30):
        draw.line((key, 0, key, np.array(img2).shape[0]), fill=128)
        previous = key
        y_list.append(key)
# print(y_list)
##creating the intermediate mising y lines
my_list = y_list
while (len(my_list) < 15):
    for i in range(len(my_list) - 1):
        j = i + 1
        diff = my_list[j] - my_list[i]
        if (diff > 100 and (i + 1) % 5 != 0):
            my_list = func(my_list, 'r', i, j)
            break
        if (diff > 220 and (i + 1) % 5 == 0):
            my_list = func(my_list, 'l', i, j)
            break
    if (len(my_list) >= 15):break
    my_list = func(my_list, 'r', len(my_list) - 1, 0)
y_list = my_list
# print(y_list)
##drawing the intermediate missing y lines ie.. the vertical lines
for i in range(len(y_list)):
    draw = ImageDraw.Draw(img2)
    draw.line((y_list[i], 0, y_list[i], np.array(img2).shape[0]), fill=128)
keys = list(row.keys())
for i in range(len(keys)):
    if int(keys[i]) > 500:
        start = int(keys[i])
        break
threshold = (int(keys[len(keys) - 1]) - int(start)) / 29
# for rows--- creating the 29 horizontal lines , note that we ignore the first 500 pixels as the answers start after that in all the test images,this ignores the question paper number of the student
previous = 0
for key, value in row.items():
    draw = ImageDraw.Draw(img2)
    if (int(key) - int(previous) > int(threshold) and value > 50 and key > 500):
        draw.line((0, key, np.array(img2).shape[1], key), fill=128)
        previous = key
        x_list.append(key)
# print("Intersection y list: ", len(y_list))
# print("Intersection x list: ", len(x_list))
img2.save(sys.argv[2])
# print("Calculating the answers from the grid system.........")
##now our grid system is ready so we create a dictionary of answers
answer_list = {}
for i in range(1, 88):answer_list[str(i)] = []
## here we check if there are atleast 15 blue pixels in a square box of 9x9 to the bottom right from the point of intersection vertical and horizontal lines and append the corresponding answwer to the list of dictionary
for i in range(len(x_list)):
    for j in range(len(y_list)):
        counter = 0
        for x in range(x_list[i], x_list[i] + 9):
            for y in range(y_list[j], y_list[j] + 9):
                r, g, b = img2.getpixel((y, x))
                if (r == 0 and g == 0 and b == 255):
                    ## get the blue pixel if there are more than 20 pixels then its the answer
                    counter += 1
                    if (counter >= 10):
                        if (j >= 0 and j <= 4):
                            key = str(i + 1)
                            if (j == 0):answer_list[key].append('A')
                            elif (j == 1):answer_list[key].append('B')
                            elif (j == 2):answer_list[key].append('C')
                            elif (j == 3):answer_list[key].append('D')
                            elif (j == 4):answer_list[key].append('E')
                        elif (j >= 5 and j <= 9):
                            key = (str(i + 1 + 29))
                            if (j == 5):answer_list[key].append('A')
                            elif (j == 6):answer_list[key].append('B')
                            elif (j == 7):answer_list[key].append('C')
                            elif (j == 8):answer_list[key].append('D')
                            elif (j == 9):answer_list[key].append('E')
                        elif (j >= 10 and j <= 14):
                            key = (str(i + 1 + 29 + 29))
                            answer_list[key]
                            if (j == 10):answer_list[key].append('A')
                            elif (j == 11):answer_list[key].append('B')
                            elif (j == 12):answer_list[key].append('C')
                            elif (j == 13):answer_list[key].append('D')
                            elif (j == 14):answer_list[key].append('E')
##writing the answers from the dictionary to the file
for key, value in answer_list.items():
    mylist = list(dict.fromkeys(value))
    answer_list[key] = mylist
# print("Final answer dictionary.........")
#print(answer_list)
file = open(sys.argv[3], "w+")
for key, value in answer_list.items():
    if(int(key)==86):break
    answer = ''
    for i in range(len(value)):answer =answer+str(value[i]) 
    #answer.strip()
    file.write(key +' '+ answer + "\n")
