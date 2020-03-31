# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:37:20 2020

@author: SUDHEER'S Dell
"""

import imageio
import math
import numpy as np

rgb=imageio.imread(r'C:/Users/Dell/Desktop/mario.jpg')
rgb=np.copy(rgb.astype(np.float32))

# RGB----------------------> XYZ
def rgbxyz(rgb):
    rgb=rgb/255.0
    for r in range(rgb.shape[0]):
        for c in range(rgb.shape[1]):
            for v in range(rgb.shape[2]):
                if rgb[r][c][v]>0.04045:
                     rgb[r][c][v]= (((rgb[r][c][v]+0.055)/1.055)**2.4) * 100
                else:
                    rgb[r][c][v]= (rgb[r][c][v]/12.92) *100
    xyz=rgb.copy()
    for r in range(xyz.shape[0]):
        for c in range(xyz.shape[1]):
            xyz[r][c][0]=rgb[r][c][0]*0.4124 +rgb[r][c][1]*0.3576 +rgb[r][c][2]*0.1805
            xyz[r][c][1]=rgb[r][c][0]*0.2126 +rgb[r][c][1]*0.7152 +rgb[r][c][2]*0.0722
            xyz[r][c][2]=rgb[r][c][0]*0.0193 +rgb[r][c][1]*0.1192 +rgb[r][c][2]*0.9505
    return xyz

# XYZ---------------------->LAB
def xyzlab(xyzf):    
    lab=xyzf.copy()
    for r in range(lab.shape[0]):
        for c in range(lab.shape[1]):
            lab[r][c][0]=xyzf[r][c][0]/95.0456
            lab[r][c][1]=xyzf[r][c][1]/100.0
            lab[r][c][2]=xyzf[r][c][2]/108.8754
    for r in range(lab.shape[0]):
        for c in range(lab.shape[1]):
            for v in range(lab.shape[2]):
                if lab[r][c][v]>0.008856:
                    lab[r][c][v]=lab[r][c][v]**(1.0/3)
                else:
                    lab[r][c][v]=(lab[r][c][v]*7.787) +(16.0/116)
    labf=lab.copy()
    for r in range(labf.shape[0]):
        for c in range(labf.shape[1]):
            labf[r][c][0]=((116*lab[r][c][1])-16)*255/100
            labf[r][c][1]=500*(lab[r][c][0]-lab[r][c][1])+128
            labf[r][c][2]=200*(lab[r][c][1]-lab[r][c][2])+128           
    return labf

# LAB-------------------->XYZ
def labxyz(labf):
    xyzc=labf.copy()
    for r in range(xyzc.shape[0]):
        for c in range(xyzc.shape[1]):
            xyzc[r][c][1]=((labf[r][c][0]*100/255)+16)/116
            xyzc[r][c][0]=(labf[r][c][1]-128)/500 +xyzc[r][c][1]
            xyzc[r][c][2]=xyzc[r][c][1]- ((labf[r][c][2]-128)/200)
            for v in range(0,3):
                if ((xyzc[r][c][v]**3)>0.008856): 
                    xyzc[r][c][v]= xyzc[r][c][v]**3
                else:
                    xyzc[r][c][v]= (xyzc[r][c][v]-(16.0/116))/7.787
    for r in range(xyzc.shape[0]):
        for c in range(xyzc.shape[1]):
            xyzc[r][c][0]=xyzc[r][c][0]*95.0456
            xyzc[r][c][1]=xyzc[r][c][1]*100.0
            xyzc[r][c][2]=xyzc[r][c][2]*108.8754        
    xyzfc=xyzc.copy()
    return xyzfc

#XYZ--------------------->RGB
def xyzrgb(xyzfc):
    xyzfc=xyzfc/100
    rgbc=xyzfc.copy()
    for r in range(xyzfc.shape[0]):
        for c in range(xyzfc.shape[1]):
            rgbc[r][c][0]=xyzfc[r][c][0]* 3.2406 +xyzfc[r][c][1]* -1.5372 +xyzfc[r][c][2]* -0.4986
            rgbc[r][c][1]=xyzfc[r][c][0]* -0.9689 +xyzfc[r][c][1]* 1.8758 +xyzfc[r][c][2]* 0.0415
            rgbc[r][c][2]=xyzfc[r][c][0]* 0.0557 +xyzfc[r][c][1]* -0.2040 +xyzfc[r][c][2]* 1.0570       
    for r in range(rgbc.shape[0]):
        for c in range(rgbc.shape[1]):
            for v in range(0,3):
                if(rgbc[r][c][v]>0.0031308):
                    rgbc[r][c][v]= 1.055*(rgbc[r][c][v]**(1/2.4)) - 0.055
                else:
                    rgbc[r][c][v]= 12.92*rgbc[r][c][v]                     
    rgbc=rgbc*255.0
    rgbcb=rgbc.astype(np.uint8) 
    return rgbcb

#-----------MEAN SHIFT------------------
xyzf=rgbxyz(rgb)
labf=xyzlab(xyzf)

for i in range(0,5):
    for x1 in range(0,labf.shape[0]):
        for y1 in range(0,labf.shape[1]):
            gl=ga=gb=c=h=0
            for x2 in range(0,labf.shape[0]):
                for y2 in range(0,labf.shape[1]):
                     mags=((x1-x2)**2 + (y1-y2)**2 )**0.5
                     magr = ((labf[x1][y1][0]-labf[x2][y2][0])**2 + (labf[x1][y1][1]-labf[x2][y2][1])**2 + (labf[x1][y1][2]-labf[x2][y2][2])**2)**(0.5)
                     if mags<=(3*7) and magr<=(3*8):
                        c+=1
                        gl+=(math.exp((-0.5)*((mags**2)/49 + (magr**2)/64))) * labf[x2][y2][0]
                        ga+=(math.exp((-0.5)*((mags**2)/49 + (magr**2)/64))) * labf[x2][y2][1] 
                        gb+=(math.exp((-0.5)*((mags**2)/49 + (magr**2)/64))) * labf[x2][y2][2] 
                        h+=(math.exp((-0.5)*((mags**2)/49 + (magr**2)/64)))
            if c>=40:
               labf[x1][y1][0]=gl/h
               labf[x1][y1][1]=ga/h
               labf[x1][y1][2]=gb/h
               
xyzfc=labxyz(labf)
rgbcb=xyzrgb(xyzfc)
#-----------MEAN SHIFT ENDS---------------


imageio.imwrite('C:/Users/Dell/Desktop/marioback.jpg',rgbcb)
