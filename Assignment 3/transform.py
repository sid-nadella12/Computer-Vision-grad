import numpy as np
import imageio
import math
import shapely
from shapely.geometry import Point, Polygon
import tqdm

def H_iter(scoord, tcoord, p, sumr):

	k = 0
	EstimatePoints = []

	while sumr > 1:
		# print(sumr, '\n')
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
		dp = np.matmul((np.linalg.inv(sumA)), sumb)
		p += dp
		# print(p.shape, H.shape)
		# H += p.flatten()
	
	H = np.append(p, 1)
	H = H.reshape((3,3))
	return H

def H_init(scoord, tcoord):

	H = np.zeros((8), dtype=float)

	sumA, sumb = np.zeros((8,8), dtype=float), np.zeros((8,1), dtype=float)
	
	#Total residual 
	sumr = 0

	I = np.array([
		[1, 0, 0],
		[0, 1, 0],
		[0, 0, 1]
	])

	for sp, tp in zip(scoord, tcoord):
		x, y= tp
		xi, yi= sp

		# H matrix
		Hc = np.append(H, 0)
		Hc = Hc.reshape((3,3))
		Hc = I + Hc

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

		# Evaluating updating A matrix
		A = np.matmul(np.transpose(jac), jac)
		sumA += A

		# residual 
		r = np.array([
			[xi-xe],
			[yi-ye]
		])
		# print(r)

		# Evaluating updating b matrix
		b = np.matmul(np.transpose(jac), r)
		sumb += b

		sumr += np.matmul(r.T, r)
	

	p = np.matmul((np.linalg.inv(sumA)), sumb)
	# print(p)
	# print(sumr)

	return p, sumr


if __name__ == '__main__':
	
	#Traget image timg
	timg = imageio.imread("samples_as3/input/target/hallway.png")
	#Target image coordinates tcoord
	tcoord = [
			(0, 492),
			(1100, 54),
			(1100, 1975),
			(0, 1225),
		]

	#Source image simg
	simg = imageio.imread("samples_as3/input/source/nerdy.png")
	#Source image coordinates scoord
	scoord = [
			(0, 0),
			(simg.shape[1], 0),
			(simg.shape[1], simg.shape[0]),
			(0, simg.shape[0])
		]

	p, sumr = H_init(scoord, tcoord)

	# For bigger image 
	#p = np.asarray([1, 0, 0, 0, 1, 0, 0, 0], dtype=float).reshape((8, 1))
	H = H_iter(scoord, tcoord, p, 10000)

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

				if src_x > src_w: src_x = src_w-1
				if src_x < 0: src_x = 0
				if src_y > src_h: src_y = src_h-1
				if src_y < 0: src_y = 0

				try:
					timg[i, j] = simg[src_y, src_x]
				except Exception as e:
					print(f'something is screwed up, i,j={i},{j} ; x,y = {src_x},{src_y}\n{e}')

	imageio.imwrite('test_out1.png', timg)