#!/usr/bin/env python3
from PIL import Image
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
import matplotlib.animation as animation

def animate_above(frame_number):
    global tx,ty,tz,xx,yy
    if(frame_number>=20):
        tz-=15

        # tx -= 5
    f = 0.002  ## camera aperture
    alpha = math.pi / 2  ##tilt angle
    beta = math.pi  ##twist angle
    gamma = math.pi  ##yaw angle
    ty+=20

    A = np.array([
        [f, 0, 0],
        [0, f, 0],
        [0, 0, 1]
    ])

    B = np.array([
        [1, 0, 0],
        [0, math.cos(alpha), -math.sin(alpha)],
        [0, math.sin(alpha), math.cos(alpha)]
    ])

    C = np.array([
        [math.cos(beta), 0, -math.sin(beta)],
        [0, 1, 0],
        [math.sin(beta), 0, math.cos(beta)]
    ])

    D = np.array([
        [math.cos(gamma), -math.sin(gamma), 0],
        [math.sin(gamma), math.cos(gamma), 0],
        [0, 0, 1]
    ])

    E = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz]
    ])

    mat2 = np.matmul(np.matmul(A, B), np.matmul(np.matmul(C, D), E))

    # tz+=0.000020
    pr,pc=[],[]
    for p in pts3:
        temp3=[[p[0]],[p[1]],[p[2]],[1]]
        temp = np.matmul(mat2,temp3)
        if(temp[2][0]>0):
            xx, yy = temp[0][0] / temp[2][0], temp[1][0] / temp[2][0]
            # print(xx,yy)
            pr+=[-xx]
            pc+=[yy]

    # print(pr)
    plt.cla()
    plt.gca().set_xlim([-0.002,0.002])
    plt.gca().set_ylim([-0.002,0.002])
    line,=plt.plot(pr,pc,'k', linestyle="", marker=".",markersize=2)
    return line,


tx,ty,tz=0,0,-10
xx,yy=1,1

with open ("airport.pts", "r") as file:
    pts3=[ [float(x) for x in l.split(" ")] for l in file.readlines()]
# pts2=[]
# for i in range(len(pts3)):
#     temp=np.matmul(mat2,np.array([[float(pts3[i][0])],[float(pts3[i][1])],[float(pts3[i][2])],[1.0000]]))
#     x,y=temp[0][0]/temp[2][0],temp[1][0]/temp[2][0]
#     pts2.append((x,y))
# print(pts2)
# (tx,ty,tz)=(0,0,5)
# (compass,tilt,twist)=(0,math.pi/2,0)

fig,ax=plt.subplots()
frame_count=50
ani=animation.FuncAnimation(fig,animate_above, frames=range(0,frame_count))

ani.save("movie.mp4")
##plt.show()
