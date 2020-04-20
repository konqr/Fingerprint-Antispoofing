import cv2 as cv
import cv2
import numpy as np
import pandas as pd
import glob
from functools import reduce
import operator

def orient_certainty(mat,sobelx,sobely):
		C = np.asarray([[0, 0],[0,0]])
		for x in range(32):
			for y in range(32):
				dx = sobelx[x,y]/(32)
				dy = sobely[x,y]/(32)
				C = C + np.asarray([[dx*dx, dy*dx],[dy*dx,dy*dy]])
		return ((C[0][0] + C[1][1] - np.sqrt((C[0][0] - C[1][1])**2 + 4*(C[0][1])**2))/(C[0][0] + C[1][1] + np.sqrt((C[0][0] - C[1][1])**2 + 4*(C[0][1])**2)))

def skeletonize(mat):
		size = np.size(mat)
		skel = np.zeros(mat.shape,np.uint8)
		ret,mat2 = cv2.threshold(mat,127,255,0)
		element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
		done = False
		c = 0
		while( not done):
		    eroded = cv2.erode(mat2,element)
		    temp = cv2.dilate(eroded,element)
		    temp = cv2.subtract(mat2,temp)
		    skel = cv2.bitwise_or(skel,temp)
		    mat2 = eroded.copy()
		    c = c+1
		    zeros = size - cv2.countNonZero(mat2)
		    if zeros==size or c>100:
		        done = True
		return skel

img_gabor_paths = glob.glob('D:\\Work\\Acad\\BTP\\data\\trainOrcathus\\*\\*\\*\\*\\*_enh.bmp')
img_enh_paths = glob.glob('D:\\Work\\Acad\\BTP\\data\\trainOrcathus\\*\\*\\*\\*\\*_crop.bmp')
minutiae_paths = glob.glob('D:\\Work\\Acad\\BTP\\data\\trainOrcathus\\*\\*\\*\\*\\*_set.csv')
# done  = pd.read_csv('D:\\Work\\Acad\\BTP\\data\\trainOrcathus\\feature.csv')
print(len(img_gabor_paths))
data = pd.DataFrame()
done = []
for n in range(len(done),len(minutiae_paths),1):
	img_gabor  = cv.imread(img_gabor_paths[n],0)
	img_enh = cv.imread(img_enh_paths[n],0)
	try:
		minutiae = pd.read_csv(minutiae_paths[n], header=None)
	except:
		continue
	#PRIMARY LEVEL TEXTURE FEATURES
	hist = cv.calcHist([img_enh],[0],None,[256],[0,256])
	#SECONDARY LEVEL TEXTURE FEATURES
	X = minutiae.iloc[:,2]
	Y = minutiae.iloc[:,1]
	patches = []
	gabor_patches = []
	ocl = []
	bw_rat = []
	fft_v = pd.DataFrame()
	fft_r = pd.DataFrame()
	for i in range(len(X)):
		patches.append(np.asarray(img_enh[X[i]-16:X[i]+16, Y[i]-16:Y[i]+16]))
		gabor_patches.append(np.asarray(img_gabor[X[i]-16:X[i]+16, Y[i]-16:Y[i]+16]))
	for i in range(len(patches)):
		mat = patches[i]
		gabor_mat = gabor_patches[i]
		if mat.shape != (32,32):
			#print(mat.shape)
			continue
		sobelx = cv.Sobel(mat,cv.CV_64F,1,0,ksize=3)
		sobely = cv.Sobel(mat,cv.CV_64F,0,1,ksize=3)
		# sobelxx = cv.Sobel(sobelx,cv.CV_64F,1,0,ksize=3)
		# sobelyy = cv.Sobel(sobely,cv.CV_64F,0,1,ksize=3)
		# sobelxy = cv.Sobel(sobelx,cv.CV_64F,0,1,ksize=3)
		# oimg = np.pi/2 + 1/2*(np.arctan2(sobelxx-sobelyy,2*sobelxy))
		# #Feature: RIDGE WIDTH
		# normal_img = np.pi/2 + oimg

		#Feature: ORIENTATION CERTAINTY
		ocl.append(orient_certainty(mat,sobelx,sobely))

		#Feature: Ridge-Valley signal extraction
		try:
			skel_valley = skeletonize(gabor_mat)
			hist_valley = cv2.calcHist([mat],[0],skel_valley,[8],[0,256])
			skel_ridge = skeletonize(255-gabor_mat)
			hist_ridge = cv2.calcHist([mat],[0],skel_ridge,[8],[0,256])
		except: 
			print("treble ", i)
			hist_valley = np.zeros((32,))
			hist_ridge = hist_valley
		#fft
		fft_v = fft_v.append([list(np.absolute(np.fft.fft(hist_valley)))],ignore_index = True)
		fft_r = fft_r.append([list(np.absolute(np.fft.fft(hist_ridge)))], ignore_index = True)
		"""np.absolute(np.fft.fft(hist_ridge))"""
		#b/w ratio
		binary = mat > 128
		bw_rat.append(np.sum(binary)/(32*32))

	#print(ocl)
	num_minutiae = len(X)	
	energy = np.sum(hist**2)
	entropy = -np.sum(hist*np.log(hist, out=np.zeros_like(hist), where=(hist!=0)))
	median = np.median(hist)
	variance = np.var(hist)
	skewness = float((pd.DataFrame(hist)).skew())
	kurt = float((pd.DataFrame(hist)).kurtosis())
	ocl_mean = np.mean(ocl)
	ocl_var = np.var(ocl)
	fft_r_mean = list(fft_r.mean(axis = 0))
	fft_v_mean = list(fft_v.mean(axis = 0))
	bw_rat_mean = np.mean(bw_rat)
	if img_enh_paths[n].find("Live") == -1:
		flag = 0
	else:
		flag = 1

	features = np.asarray(([flag, num_minutiae, energy, entropy, median, variance,skewness,kurt,ocl_mean, ocl_var,bw_rat_mean]+fft_v_mean+fft_r_mean)).ravel()
	data = data.append([list(features)], ignore_index=True)
	print(n, '\r', end = '')
		
data.to_csv('D:\\Work\\Acad\\BTP\\data\\trainOrcathus\\feature.csv', mode='a', header=False)

"""cv2.imshow('ImageWindow', img); cv2.waitKey()"""