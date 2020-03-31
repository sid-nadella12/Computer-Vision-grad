import sys, imageio, numpy as np, math, os
from tqdm import tqdm
np.set_printoptions(suppress=True)

def pointEstimate(left_pixel_x, left_pixel_y, 
		left_intrinsic_K, left_extrinsic_R, left_extrinsic_t,
		right_pixel_x, right_pixel_y,
		right_intrinsic_K, right_extrinsic_R, right_extrinsic_t):
	# left view vj, cj
	left_vj = (left_extrinsic_R.T @ 
		np.linalg.inv(left_intrinsic_K) @ 
		np.vstack([left_pixel_x, left_pixel_y, 1]))
	left_cj = -(left_extrinsic_R.T @
		left_extrinsic_t)

	# right view vj, cj
	right_vj = (right_extrinsic_R.T @ 
		np.linalg.inv(right_intrinsic_K) @ 
		np.vstack([right_pixel_x, right_pixel_y, 1]))
	right_cj = -(right_extrinsic_R.T @
		right_extrinsic_t)

	# convert vj values into unit vectors
	left_vj = left_vj / np.linalg.norm(left_vj)
	right_vj = right_vj / np.linalg.norm(right_vj)
	# print(left_vj, left_cj, right_vj, right_cj)
	
	# create accumulators
	left_expression_accumulator = np.zeros((3,3))
	right_expression_accumulator = np.zeros((3,1))

	# vj and cj lists to be zipped in for loop
	list_vj = list((left_vj, right_vj))
	list_cj = list((left_cj, right_cj))

	for vj, cj in zip(list_vj, list_cj):
		left_expression_accumulator += (
			np.identity(3)-vj@vj.T)
		right_expression_accumulator += (
			(np.identity(3)-vj@vj.T)@cj)

	# invert left expression accumulator
	left_expression_accumulator = np.linalg.inv(left_expression_accumulator)

	prediction = left_expression_accumulator @ right_expression_accumulator
	return prediction

def GraphicsCalculation(x, y, z):
	# first attempt
	# # left view
	# left_intrinsic_K = np.array([
	# 	[0.63261207, 0.00000000, 258.00000000],
	# 	[0.00000000, 0.63261207, 204.00000000],
	# 	[0.00000000, 0.00000000, 1.00000000]])
	# left_extrinsic_Rt = np.array([
	# 	[-0.64210404, -0.45190266, 0.61926278, -3.94567011],
	# 	[0.18736405, 0.69078066, 0.69836723, -1.64165817],
	# 	[-0.74336876, 0.56445200, -0.35888274, 0.84689138]])
	# # right view
	# right_intrinsic_K = np.array([
	# 	[0.35128280, 0.00000000, 258.00000000],
	# 	[0.00000000, 0.35128280, 204.00000000],
	# 	[0.00000000, 0.00000000, 1.00000000]])
	# right_extrinsic_Rt = np.array([
	# 	[0.73456601, -0.61652611, -0.28338725, 0.28892156],
	# 	[0.67467495, 0.61913764, 0.40184859, -1.32876651],
	# 	[-0.07229443, -0.48637859, 0.87075219, -1.06300875]])
	
	# third Tsai calibration attempt
	# # left view
	# left_intrinsic_K = np.array([
	# 	[0.00987113, 0.00000000, 2688.00000000],
	# 	[0.00000000, 0.00987113, 2016.00000000],
	# 	[0.00000000, 0.00000000, 1.00000000]])
	# left_extrinsic_Rt = np.array([
	# 	[0.20416376, 0.25383456, -0.94545501, 7.92292300],  
	# 	[0.50334732, 0.80112914, 0.32378013, -1.84132963],
	# 	[0.83961815, -0.54199641, 0.03579460, 0.27078827]])
	# # right view
	# right_intrinsic_K = np.array([
	# 	[-0.00947067, 0.00000000, 2688.00000000],
	# 	[0.00000000, -0.00947067, 2016.00000000],
	# 	[0.00000000, 0.00000000, 1.00000000]])
	# right_extrinsic_Rt = np.array([
	# 	[0.75913513, 0.61292654, -0.21916870, 5.16855531],
	# 	[0.18708703, -0.52793919, -0.82841937, 1.59405234],
	# 	[-0.62346796, 0.58787862, -0.51544779, 0.91946724]])

	# synthetic Tsai calibration attempt
	# left view
	left_intrinsic_K = np.array([
		[0.50000000, 0.00000000, 1500.00000000],
		[0.00000000, 0.50000000, 2000.00000000],
		[0.00000000, 0.00000000, 1.00000000]])
	left_extrinsic_Rt = np.array([
		[0.20416376, 0.25383456, -0.94545501, 7.92292300],  
		[0.50334732, 0.80112914, 0.32378013, -1.84132963],
		[0.83961815, -0.54199641, 0.03579460, 0.27078827]])
	# right view
	right_intrinsic_K = np.array([
		[0.50000000, 0.00000000, 1500.00000000],
		[0.00000000, 0.50000000, 2000.00000000],
		[0.00000000, 0.00000000, 1.00000000]])
	right_extrinsic_Rt = np.array([
		[0.75913513, 0.61292654, -0.21916870, 5.16855531],
		[0.18708703, -0.52793919, -0.82841937, 1.59405234],
		[-0.62346796, 0.58787862, -0.51544779, 0.91946724]])

	left_world_p = np.vstack([x, y, z, 1])
	left_image_x = left_intrinsic_K@left_extrinsic_Rt@left_world_p
	right_world_p = np.vstack([x, y, z, 1])
	right_image_x = right_intrinsic_K@right_extrinsic_Rt@right_world_p
	return np.array([
			left_image_x[0]/left_image_x[2], 
			left_image_x[1]/left_image_x[2],
			right_image_x[0]/right_image_x[2], 
			right_image_x[1]/right_image_x[2]])

def main(left_pixel_x=1408, left_pixel_y=759,
	right_pixel_x=1594, right_pixel_y=828):
	# K, R, and t matrixes from Tsai camera calibration
	# first Tsai calibration attempt
	# left view
	left_intrinsic_K = np.array([
		[0.63261207, 0.00000000, 258.00000000],
		[0.00000000, 0.63261207, 204.00000000],
		[0.00000000, 0.00000000, 1.00000000]])
	left_extrinsic_R = np.array([
		[-0.64210404, -0.45190266, 0.61926278],
		[0.18736405, 0.69078066, 0.69836723],
		[-0.74336876, 0.56445200, -0.35888274]])
	left_extrinsic_t = np.array([
		[-3.94567011],
		[-1.64165817],
		[0.84689138]])
	# right view
	right_intrinsic_K = np.array([
		[0.35128280, 0.00000000, 258.00000000],
		[0.00000000, 0.35128280, 204.00000000],
		[0.00000000, 0.00000000, 1.00000000]])
	right_extrinsic_R = np.array([
		[0.73456601, -0.61652611, -0.28338725],
		[0.67467495, 0.61913764, 0.40184859],
		[-0.07229443, -0.48637859, 0.87075219]])
	right_extrinsic_t = np.array([
		[0.28892156],
		[-1.32876651],
		[-1.06300875]])
	# second Tsai calibration attempt
	# 	# left view
	# left_intrinsic_K = np.array([
	# 	[1450.5407505053759, 0.00000000, 3024/2],
	# 	[0.00000000, 1450.5407505053759, 4032/2],
	# 	[0.00000000, 0.00000000, 1.00000000]])
	# left_extrinsic_R = np.array([
	# 	[-0.6161174313398848,
 #         0.7851491152755808,
 #         -0.06277083384116344],
 #        [0.5183729844763256,
 #         0.4641930670063725,
 #         0.7182020923864845],
 #        [0.5930335233055963,
 #         0.4099581238677801,
 #         -0.6929975302340987]])
	# left_extrinsic_t = np.vstack(
	# 	[-0.68700767, -2.53984893,  2.57909975])
	# # right view
	# right_intrinsic_K = np.array([
	# 	[584.010947926578, 0.00000000, 3024/2],
	# 	[0.00000000, 584.010947926578, 4032/2],
	# 	[0.00000000, 0.00000000, 1.00000000]])
	# right_extrinsic_R = np.array([
	# 	[-0.7284188300422267,
 #         0.6799623764965412,
 #         -0.0840069913107796],
 #        [0.32440359549450565,
 #         0.4502955111158061,
 #         0.8318631256998907],
 #        [0.6034635989603273,
 #         0.5786925947496916,
 #         -0.5485859691167088]])
	# right_extrinsic_t = np.vstack(
	# 	[ 0.27361856, -2.56014888,  0.84176459])
	# third Tsai calibration attempt
	# # left view
	# left_intrinsic_K = np.array([
	# 	[0.00987113, 0.00000000, 2688.00000000],
	# 	[0.00000000, 0.00987113, 2016.00000000],
	# 	[0.00000000, 0.00000000, 1.00000000]])
	# left_extrinsic_R = np.array([
	# 	[0.20416376, 0.25383456, -0.94545501],  
	# 	[0.50334732, 0.80112914, 0.32378013],
	# 	[0.83961815, -0.54199641, 0.03579460]])
	# left_extrinsic_t = np.vstack(
	# 	[7.92292300, -1.84132963,  0.27078827])
	# # right view
	# right_intrinsic_K = np.array([
	# 	[-0.00947067, 0.00000000, 2688.00000000],
	# 	[0.00000000, -0.00947067, 2016.00000000],
	# 	[0.00000000, 0.00000000, 1.00000000]])
	# right_extrinsic_R = np.array([
	# 	[0.75913513, 0.61292654, -0.21916870],
	# 	[0.18708703, -0.52793919, -0.82841937],
	# 	[-0.62346796, 0.58787862, -0.51544779]])
	# right_extrinsic_t = np.vstack(
	# 	[5.16855531, 1.59405234, 0.91946724])

	# print(left_intrinsic_K.shape,left_extrinsic_R.shape,left_extrinsic_t.shape,
	# 	right_intrinsic_K.shape,right_extrinsic_R.shape,right_extrinsic_t.shape)
	worldCoordinate = pointEstimate(
						left_pixel_x, left_pixel_y,
						left_intrinsic_K, left_extrinsic_R, left_extrinsic_t,
						right_pixel_x, right_pixel_y,
						right_intrinsic_K, right_extrinsic_R, right_extrinsic_t)
	print(worldCoordinate)
	
if __name__ == '__main__':
	# points used for calibration
	# main(2909,1217,2776,1354) # top face, right (3,0,3)
	# main(147,1363,172,1300) # top face, left (0,3,3)
	# main(2185,2785,1758,2879) # right face, bottom left (1,0,1)

	# points used for testing
	# main(1408,759,1594,828) # top face, back (3,3,3)

	# test functionality of triangulation code
	# worldCoordinateList = ((3,0,3),(0,3,3),(0,1,2),(0,2,1),(2,0,2),(1,0,1))
	worldCoordinateList = ((3,0,3),(0,3,3),(0,1,2),(0,2,1),(2,0,2),(1,0,1))
	print('left')
	for x,y,z in worldCoordinateList:
		xj_leftRight = GraphicsCalculation(x,y,z)
		print('{} {} {} {} {}'.format(x,y,z,float(xj_leftRight[0]),float(xj_leftRight[1])))
		# main(xj_leftRight[0],xj_leftRight[1],xj_leftRight[2],xj_leftRight[3])
	print('right')
	for x,y,z in worldCoordinateList:
		xj_leftRight = GraphicsCalculation(x,y,z)
		print('{} {} {} {} {}'.format(x,y,z,float(xj_leftRight[2]),float(xj_leftRight[3])))
		# main(xj_leftRight[0],xj_leftRight[1],xj_leftRight[2],xj_leftRight[3])