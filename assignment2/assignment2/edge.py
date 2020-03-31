# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 04:44:34 2020

@author: Dell SID NADELLA
"""

import pandas as pd
import numpy as np
import imageio
import math
import os
    
def firstOrderEdgeDetection(img, sigma):
    d = 1*sigma
    kern = np.zeros((2*d+1), dtype=float)
    #
    # First derivative of h(x)
    #
    #  -x * exp-(x**2/2*(sigma**2))  = num
    #   ---------------------------
    #    (sqrt(2*pi))(sigma**3)      = den
    #
    den = math.sqrt(2.0 * math.pi) * math.pow(sigma, 3)
    for i in range(2*d+1):
        x = i - d
        num = -1.0*x*math.exp(-1*x**2/(2*(sigma**2)))
        kern[i] = num / den
    #print(kern)
    
    # Now we will set dx value computing the product of padded matrix and the kern
    # We have to pad around the borders of the image as we will map the reference frame over the pixels
    padx = np.pad(img, [(d, d), (d, d)], mode='constant')
    #pad.shape
    dx=img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            dx[i][j]=0.0
    #print(dx)
    
    rows= padx.shape[0]
    colm= padx.shape[1]
    for i in range(d, rows-d):
        for j in range(d, colm-d):
            # setting window frame
            beg, end = j-d, j+d
            #print(pad[i,start:stop+1])
            #Calculating the pixel values of dx 
            total = np.multiply(padx[i, beg:end+1], kern)
            dx[i-d, j-d]= np.sum(total)
    #dx
    
    ## Now compute dy
    pady = np.pad(img, [(d, d), (d, d)], mode='constant')
    pady = np.transpose(pady)
    dy=pady.copy()
    for i in range(pady.shape[0]):
        for j in range(pady.shape[1]):
            dy[i][j]=0.0
    #print(dy)
    
    row= pady.shape[0]
    col= pady.shape[1]
    for i in range(d, row-d):
        for j in range(d, col-d):
            # setting window frame
            beg, end = j-d, j+d
            #print(pad[i,start:stop+1])
            #Calculating the pixel values of dx 
            total = np.multiply(pady[i, beg:end+1], kern)
            dy[i-d, j-d]= np.sum(total)
            
    #dy
    return dx, np.transpose(dy)

def zeroOrderDetection(dx, dy, sigma, K):	
	
    dist = 1 * sigma

    # creating eigen array just like kern
    eigen = np.zeros(dx.shape, dtype=float)

    # Creating kernel for o order guassian 
    kern = np.zeros((2*dist+1, 2*dist+1), dtype=float)
    #
    #               1 * exp-((x**2 + y**2)/(2*sigma**2))
    #   h(x,y)  =  --------------------------------------
    #               (2.pi)*sigma**2
    #
    #   const= 1/((2.pi)*sigma**2)
    #
    const = 1.0/((2.0*math.pi)/sigma**2)
    for i in range(2*dist+1):
        for j in range(2*dist+1):
            x, y = i-dist, j-dist
            kern[i,j] = const*math.exp(-1.0*(x**2 + y**2)/(2*sigma**2))

	# Pad the image and calculate the eigen values
    padx = np.pad(dx, [(dist, dist), (dist, dist)], 'constant')
    pady = np.pad(dy, [(dist, dist), (dist, dist)], 'constant')
    rows, cols = padx.shape
    for i in range(dist, rows-dist):
        for j in range(dist, cols-dist):
            # setting window frame
            begx, begy = i-dist, j-dist
            endx, endy = i+dist, j+dist

            # calculating 
            dxdx = np.square(padx[begx:endx+1, begy:endy+1])
            IxIx = np.sum(np.multiply(dxdx, kern))
        
            dydy = np.square(pady[begx:endx+1, begy:endy+1])
            IyIy = np.sum(np.multiply(dydy, kern))
            
            dxdy = np.multiply(padx[begx:endx+1, begy:endy+1], pady[begx:endx+1, begy:endy+1])
            IxIy = np.sum(np.multiply(dxdy, kern))

            # Matrix A
            # [IxIx  IxIy ]
            # [IyIx  IyIy ]
            A = np.array([[IxIx, IxIy], [IxIy, IyIy]])
            #print(A)
            # numpy.linalg.eigvals(a): computes eigen values of general matrix a 
            eigen[i-dist,j-dist] = np.min(np.linalg.eigvals(A))

    # Find maximum of the frames
    eigenMax = np.copy(eigen)
    rows, cols = eigen.shape
    for i in range(rows):
        for j in range(cols):
            # Sets boundaries
            begx, begy, endx, endy = i-dist, j-dist, i+dist, j+dist

            if begx < 0: begx = 0
            if begy < 0: begy = 0
            if endx > rows-1: endx = rows-1
            if endy > cols-1: endy = cols-1

            # Calculate the maximum and check wether the current is maximum or not
            max = np.max(eigen[begx:endx+1, begy:endy+1])
            if eigen[i,j] < max : eigenMax[i,j] = 0

	# Findout top K eigen values
    points = np.zeros(eigenMax.shape, dtype=np.uint8)
    xpoints, ypoints = np.unravel_index(eigenMax.flatten().argsort()[-K:], eigenMax.shape)

    # Add those top K points in a list
    pointList = []
    for x, y in zip(xpoints, ypoints):
        points[x,y] = eigenMax[x,y]
        pointList.append((x, y))

    # Return matrix and pointlist.
    return pointList, points, xpoints, ypoints

def correlate(imgprev, imgcurrent, xp, yp, sigma):
    xpointsn=[]
    ypointsn=[]
    newx=0
    newy=0
    d=sigma
    for fx, fy in zip(xp, yp):
        #                    Kernel
        #               1 * exp-((x**2 + y**2)/(2*sigma**2))
        #   h(x,y)  =  --------------------------------------
        #               (2.pi)*sigma**2
        kern = np.zeros((d, d), dtype=float)
        denominator = 1.0/(2.0*math.pi*(sigma**2))
        for i in range(d):
            for j in range(d):
                x, y = i-d, j-d
                kern[i,j] = denominator*math.exp(-1.0*(x**2 + y**2)/(2*sigma**2))
        #print(kern)

        #Image patch of previous Image. Patch size= 3*3
        i=0
        j=0
        pm = np.zeros((3, 3), dtype=float)
        for bx in range(fx-1, fx+2):
            for by in range(fy-1, fy+2):
                if bx<0: bx=0
                if by<0: by=0
                if bx>len(imgprev)-1: bx=len(imgprev)-1
                if by>len(imgprev[0])-1: by=len(imgprev[0])-1
                #print(imgprev[bx][by])
                pm[i][j]=imgprev[bx][by]
                j+=1
            i+=1
            j=0
        #print(pm)#prev matrix

        #Image patch of current image
        cm=np.zeros((3, 3),dtype=float) 
        total=[]
        # sum matrix is the result of SSD (Sum of Square Difference)
        sum=np.zeros((15,15),dtype=float)
        ir=0
        ij=0
        for wx in range(fx-7,fx+8): #window size 15*15
            for wy in range(fy-7,fy+8):
                # Boundary checks
                if wx<0: wx=0
                if wy<0: wy=0
                if wx>len(imgprev)-1: wx=len(imgprev)-1
                if wy>len(imgprev[0])-1: wy=len(imgprev[0])-1
                i1=0
                j1=0
                # Current image patch. Patch size 3*3
                for bx in range(wx-1, wx+2):
                    for by in range(wy-1, wy+2):
                        # Boundary check
                        if bx<0: bx=0
                        if by<0: by=0
                        if bx>len(imgprev)-1: bx=len(imgprev)-1
                        if by>len(imgprev[0])-1: by=len(imgprev[0])-1
                        cm[i1][j1]=imgcurrent[bx][by]
                        j1+=1
                    i1+=1
                    j1=0
                #  Ewssd(u) = sum( W(xi)* ([I(xi + u)-I(xi)]^2) )
                #  Find out square of currentPatch - prevPatch
                #  Multiply it with Kernel
                #  Perform the sum of result matrix assign the value to sum matrix  
                difkern=np.multiply(np.square(np.subtract(cm,pm)),kern)
                #print(difkern)
                # Store the value in sum matrix
                sum[ir][ij]=np.sum(difkern)
                ij+=1
            ir+=1
            ij=0

        # The min of the SSD's is our new point    
        nx, ny = np.unravel_index(np.argmin(sum), sum.shape)

        # Get the location of the pixel 
        newx=fx-7+nx
        # Boundary check
        if newx<0:
            newx=0
        if newx>len(imgprev)-1:
            newx=len(imgprev)-1
        #print(newx)

        newy=fy-7+ny
        # Boundary check
        if newy<0:
            newy=0
        if newy>len(imgprev[0])-1:
            newy=len(imgprev[0])-1

        # Update the new feature points list    
        xpointsn.append(newx)
        ypointsn.append(newy)
    return xpointsn, ypointsn

if __name__ == '__main__':

    # Read the first image
    img = imageio.imread('C:/Users/Dell/Desktop/spring 2020/cv/assignment2/samples/samples/sample-input/walking_frames/22.jpg',as_gray=True)
    imgcolor= imageio.imread('C:/Users/Dell/Desktop/spring 2020/cv/assignment2/samples/samples/sample-input/walking_frames/22.jpg')
    #print(img)
    sigma=3
    dx, dy= firstOrderEdgeDetection(img, sigma)
    imageio.imwrite('C:/Users/Dell/Desktop/spring 2020/cv/assignment2/samples/samples/output-generated/walking_frames/0dx.png', dx)
    imageio.imwrite('C:/Users/Dell/Desktop/spring 2020/cv/assignment2/samples/samples/output-generated/walking_frames/0dy.png', dy)
    k=7 #Feature points
    pts, imgeigen, xprevpoints, yprevpoints = zeroOrderDetection(dx, dy, sigma, k)

    #  Mark the feature points by RED  
    for i in range(0,len(pts)):
        imgcolor[pts[i][0]][pts[i][1]][0]=255
        imgcolor[pts[i][0]][pts[i][1]][1]=0
        imgcolor[pts[i][0]][pts[i][1]][2]=0
    imageio.imwrite('C:/Users/Dell/Desktop/spring 2020/cv/assignment2/samples/samples/output-generated/walking_frames/0.png', imgcolor)
    
    # Track the feature points in the subsequent frames of the video. 
    for f in range(23,): 
        
        imgnew = imageio.imread('C:/Users/Dell/Desktop/spring 2020/cv/assignment2/samples/samples/sample-input/walking_frames/'+str(f)+'.jpg',as_gray=True)
        imgnewcolor= imageio.imread('C:/Users/Dell/Desktop/spring 2020/cv/assignment2/samples/samples/sample-input/walking_frames/'+str(f)+'.jpg')
        xnewpoints, ynewpoints= correlate(imgeigen, imgnew, xprevpoints, yprevpoints, sigma)
        
        # Mark the feature points by RED
        for i,j in zip(xnewpoints,ynewpoints):
            imgnewcolor[i][j][0]=255
            imgnewcolor[i][j][1]=0
            imgnewcolor[i][j][2]=0
        imageio.imwrite('C:/Users/Dell/Desktop/spring 2020/cv/assignment2/samples/samples/output-generated/walking_frames/'+str(f)+'.png', imgnewcolor)
        imgeigen=np.copy(imgnew)

        #update the new points to previous points 
        xprevpoints=xnewpoints.copy()
        yprevpoints=ynewpoints.copy()
            