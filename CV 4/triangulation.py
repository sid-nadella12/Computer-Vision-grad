import numpy as np
from scipy import linalg
from numpy.linalg import lstsq
from numpy.linalg import qr
from numpy import arctan

# Tsai 
def Tsai(datafile):

	def A_matrix(X, x):
    # Build the 2*12 A_i matrix given the X and x values i.e, the 3-D and 2-D poim
	    A = np.zeros((2, 12))
	    x1,x2,x3 = x[0], x[1], x[2]
	    X1,X2,X3,X4 = X[0], X[1], X[2], X[3]
	    A[0] = [0, -x3*X1, x2*X1, 0, -x3*X2, x2*X2, 0, -x3*X3, x2*X3, 0, -x3*X4, x2*X4]
	    A[1] = [x3*X1, 0, -x1*X1, x3*X2, 0, -x1*X2, x3*X3, 0, -x1*X3, x3*X4, 0, -x1*X4]
	    return A

	def rq(M):
	    # Compute the RQ decomposition of the matrix M from the numpy.linalg.qr decomposition
	    n, _ = M.shape
	    J = np.fliplr(np.eye(n))
	    q, r = qr(np.dot(J, np.dot(M.T, J)))
	    return np.dot(J, np.dot(r.T, J)), np.dot(J, np.dot(q.T, J))

	def mat_P(p):
	    # Reshaping the P of size 12 into the corresponding 3*4 P matrix
	    matP = np.zeros((3, 4))
	    matP[0] = [p[0], p[3], p[6], p[9]]
	    matP[1] = [p[1], p[4], p[7], p[10]]
	    matP[2] = [p[2], p[5], p[8], p[11]]
	    return matP

	def mat_K(p):
	    # Compute the matrix K from the vector p of size 12
	    r, q = rq(mat_P(p)[:, :3])
	    return r/r[2, 2]

	data = np.loadtxt(datafile)

	X = data[:, :3] # 3D points of the calibration rig
	x = data[:, 3:] # Observed 2D points

	n, _ = X.shape
	ones = np.ones((n, 1))

	# Adding an extra 1 coordinate to have normalized points.
	X = np.concatenate((X, ones), axis=1)
	x = np.concatenate((x, ones), axis=1)

	A = np.zeros((0, 12))
	for i in range(n):
	    A = np.concatenate((A, A_matrix(X[i, :], x[i, :])))

	# Solving A_11x = b with A_11 the A matrix without the last column and x a vector of size 11
	m = lstsq(A[:, :11], -A[:, -1], rcond=None)
	p = m[0]
	# print('p')
	# print(p)
	p = np.append(p, 1)
	K = mat_K(p)
	P = mat_P(p)
	# print('\nProjection matrix P :\n{}'.format(P))
	# print('\nCalibration matrix K :\n{}'.format(K))
	return P, K

# Triangulation	
def triangulation(K_left, R_left, T_left, K_right, R_right, T_right, left_view_pixel_x, left_view_pixel_y, right_view_pixel_x, right_view_pixel_y):
	
	# Point on Image 1(left view)
	point_left = np.array([
		[left_view_pixel_x],
		[left_view_pixel_y],
		[1]
		])

	# Point on Image 2(right view)
	point_right = np.array([
		[right_view_pixel_x],
		[right_view_pixel_y],
		[1]
		])

	T_left = np.array([
		[T_left[0]],
		[T_left[1]],
		[T_left[2]]
		])

	T_right = np.array([
		[T_right[0]],
		[T_right[1]],
		[T_right[2]]
		])

	# calculating Cj for Both the cameras:
	Cj1_left = calculate_Cj(R_left, T_left)
	Cj1_right = calculate_Cj(R_right, T_right)

	# calculating the Vj for both the cameras:
	Vj1_left = calculate_Vj(R_left, K_left, point_left)
	Vj1_right = calculate_Vj(R_left, K_left, point_right)

	P1 = P_estimate(Cj1_left, Cj1_right, Vj1_left, Vj1_right)
	print(P1)

# Esimating P
def P_estimate(Cj_left, Cj_right, Vj_left, Vj_right):
	#Identity matrix
	I = np.identity(3)
	Vj_sum = np.zeros((3,3))
	Vj_sum_inverse = np.zeros((3,3))
	Cj_sum = np.zeros((3,1))
	P_cal = np.zeros((3,1))

	Vj_sum = (I - np.matmul(Vj_left, Vj_left.T)) + (I - np.matmul(Vj_right, Vj_right.T))
	Cj_sum = np.matmul((I - np.matmul(Vj_left, Vj_left.T)), Cj_left) + np.matmul((I - np.matmul(Vj_right, Vj_right.T)), Cj_right)

	# Vj_sum_inverse = np.linalg.inv(Vj_sum)

	P_cal = np.matmul((np.linalg.inv(Vj_sum)), Cj_sum)

	return P_cal

# calculate Cj 
def calculate_Cj(R, T):
	return -1*np.matmul(np.linalg.inv(R), (T))
	# return -1*np.matmul(R.T, T)

# calclate Vj
def calculate_Vj(R, K, point):
	# vj = np.matmul(R.T, np.matmul(np.linalg.inv(K), point))
	vj = -1 * np.matmul(np.linalg.inv(R), np.matmul(np.linalg.inv(K), point))
	vj = vj / linalg.norm(vj)
	return vj

# Getting the 2D co-ordinates from 3D location
def reverse_calc(X, Y, Z, P_left, K_left, P_right, K_right):

	RT_cam1 = np.matmul(np.linalg.inv(K_left), P_left)
	R_left = RT_cam1[:, :3]
	T_left = RT_cam1[:, 3] 	

	RT_cam2 = np.matmul(np.linalg.inv(K_right), P_right)
	R_right = RT_cam2[:, :3]
	T_right = RT_cam2[:, 3] 

	P_world = np.vstack([X, Y, Z, 1])
	left_image_x = P_left@P_world
	right_image_x = P_right@P_world

	pixel_coord = np.array([
		[left_image_x[0] / left_image_x[2]],
		[left_image_x[1] / left_image_x[2]],
		[right_image_x[0] / right_image_x[2]],
		[right_image_x[1] / right_image_x[2]]
		])
	# print(pixel_coord)

	return K_left, R_left, T_left, K_right, R_right, T_right

# Main
if __name__ == '__main__':

	Data_left = 'data_left.txt'
	Data_right = 'data_right.txt'
	
	P_left, K_left = Tsai(Data_left)
	P_right, K_right = Tsai(Data_right)

	K_left, R_left, T_left, K_right, R_right, T_right = reverse_calc(1, 1, 0, P_left, K_left, P_right, K_right)
	triangulation(K_left, R_left, T_left, K_right, R_right, T_right, 2185, 2785, 1758, 2879); #1 1 0
	print(' ')
	K_left, R_left, T_left, K_right, R_right, T_right = reverse_calc(0, 3, 0, P_left, K_left, P_right, K_right)
	triangulation(K_left, R_left, T_left, K_right, R_right, T_right, 2026, 2262, 987, 2250); #0 3 0

		
