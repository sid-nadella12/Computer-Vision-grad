import numpy as np
import imageio
import math
import shapely
from shapely.geometry import Point, Polygon
import tqdm

def H_iter(scoord, tcoord, p, sumr, dampfac):

	k = 0
	EstimatePoints = []
	sumr = 9999999999999
	sumprev = sumr + 1
	while sumr < sumprev :
		# print(sumr, '\n')
		sumprev = sumr
		sumr = 0
		sumA, sumb = np.zeros((8,8)), np.zeros((8,1))
		for sp, tp in zip(scoord, tcoord):
			x, y= tp
			xi, yi= sp

			# H matrix
			Hc = np.append(p, 1)
			Hc = Hc.reshape((3,3))
			# Hc = I + Hc

			pt = np.array([
				[x],
				[y],
				[1]
			])

			abD = np.matmul(Hc, pt)
			D = 1.0/ abD[2,0]
			# x estimate and y estimate
			xe = abD[0,0]*D
			ye = abD[1,0]*D

			xeneg = -1.0*xe
			yeneg = -1.0*ye

			jac = np.array([
				[x, y, 1, 0, 0, 0, x*xeneg, y*xeneg],
				[0, 0, 0, x, y, 1, x*yeneg, y*yeneg]
			])*D

			# residual 
			r = np.array([
				[xi-xe],
				[yi-ye]
			])

			print(r, '\n')

			# Evaluating A matrix and updating A matrix
			A = np.matmul(np.transpose(jac), jac)
			sumA += A

			# Evaluating b matrix and updating b matrix
			b = np.matmul(np.transpose(jac), r)
			sumb += b

			sumr += np.matmul(r.T, r)


		
		# print(sumA, sumb)
		# print(f'sumr = {sumr}')
		diagA = np.diag(np.diag(sumA))

		dp = np.matmul(np.linalg.inv(sumA + dampfac*diagA), sumb)

		p += dp
		# print(p.shape, H.shape)
		# H += p.flatten()
	
	H = np.append(p, 1)
	H = H.reshape((3,3))
	return H

def H(scoord, tcoord):

	Xtar = np.asarray(tcoord).T
	Xtar = np.append(Xtar, [[1]*len(tcoord)], axis=0)

	Xsrc = np.asarray(scoord).T
	Xsrc = np.append(Xsrc, [[1]*len(scoord)], axis=0)

	I1 = np.asarray([[1, 0, 0],[0, 1, 0]])
	delZ = Xsrc - Xtar
	delZ = np.matmul(I1, delZ)
	delX = np.append(np.vstack(delZ[0]), np.vstack(delZ[1]), axis=0)

	p1= np.asarray([
		[0, 0, 1, 0, 0, 0],
		[0, 0, 0, 1, 0, 0],
		[1, 0, 0, 0, 0, 0]
	])
	p2 = np.asarray([
		[0, 0, 0, 0, 1, 0],
		[0, 0, 0, 0, 0, 1],
		[0, 1, 0, 0, 0, 0]
	])

	j1 = np.matmul(np.transpose(Xtar), p1)
	j2 = np.matmul(np.transpose(Xtar), p2)

	# J matrix
	J = np.vstack([j1, j2])
	
	# A matrix
	A = np.matmul(np.transpose(J), J)
	
	# b matirx
	b = np.matmul(np.transpose(J), delX)

	# p matrix
	p = np.matmul(np.linalg.inv(A), b)
	p = np.append(p, np.vstack([0, 0]), axis=0)

	return p


if __name__ == '__main__':
	
	#Traget image timg
	timg = imageio.imread("samples_as3/input/target/hallway.png")
	#Target image coordinates tcoord
	tcoord =[
			(0, 492),
			(1100, 54),
			(1100, 1975),
			(0, 1225)
		]

	#Source image simg
	simg = imageio.imread("samples_as3/input/source/nerdy.png")
	#Source image coordinates scoord
	scoord = [
			(0, 0),
			(599, 0),
			(599, 349),
			(0, 349)
    	]

	p= H(scoord, tcoord)

	# For bigger image 
	#p = np.asarray([1, 0, 0, 0, 1, 0, 0, 0], dtype=float).reshape((8, 1))
	H = H_iter(scoord, tcoord, p, 1000, 0.1)

	src_h = simg.shape[0]
	src_w = simg.shape[1]
	poly = Polygon(tcoord)
	for i in tqdm.tqdm(range(timg.shape[0])):
		for j in range(timg.shape[1]):
			pp = Point(j, i)

			if pp.within(poly):
				
				src_xy_vec = np.matmul(H, np.vstack([j, i, 1]))
				src_x = int(src_xy_vec[0] / src_xy_vec[2])
				src_y = int(src_xy_vec[1] / src_xy_vec[2])

				# print(f'src_x, src_y = {src_x}, {src_y}')

				if src_x > src_w: src_x = src_w-2
				if src_x < 0: src_x = 0
				if src_y > src_h: src_y = src_h-2
				if src_y < 0: src_y = 0

				try:
					timg[i, j] = simg[src_y, src_x]
				except Exception as e:
					print(f'something is screwed up, i,j={i},{j} ; x,y = {src_x},{src_y}\n{e}')

	imageio.imwrite('test_outhalllonerdy.png', timg)