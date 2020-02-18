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

img_paths = glob.glob('D:\\Work\\Acad\\BTP\\data\\testDigitalPersona\\*\\*\\*\\*.png')
img_enh_paths = glob.glob('D:\\Work\\Acad\\BTP\\data\\testDigitalPersona\\*\\*\\*\\*_crop.bmp')
minutiae_paths = glob.glob('D:\\Work\\Acad\\BTP\\data\\testDigitalPersona\\*\\*\\*\\*_set.csv')
# done  = pd.read_csv('D:\\Work\\Acad\\BTP\\data\\trainGreenBit\\feature.csv')
data = pd.DataFrame()
done = []
for n in range(len(done),len(minutiae_paths),1):
	img  = cv.imread(img_paths[n],0)
	img_enh = cv.imread(img_enh_paths[n],0)
	try:
		minutiae = pd.read_csv(minutiae_paths[n], header=None)
	except:
		continue
	#PRIMARY LEVEL TEXTURE FEATURES
	hist = cv.calcHist([img],[0],None,[256],[0,256])

	#SECONDARY LEVEL TEXTURE FEATURES
	X = minutiae.iloc[:,2]
	Y = minutiae.iloc[:,1]
	patches = []
	ocl = []
	bw_rat = []
	fft_v = pd.DataFrame()
	fft_r = pd.DataFrame()
	for i in range(len(X)):
		patches.append(np.asarray(img_enh[X[i]-16:X[i]+16, Y[i]-16:Y[i]+16]))
	for i in range(len(patches)):
		mat = patches[i]
		if mat.shape != (32,32):
			#print(mat.shape)
			continue
		binary = mat > 128
		bw_rat.append(np.sum(binary)/(32*32))
	bw_rat_mean = np.mean(bw_rat)
	data = data.append([bw_rat_mean], ignore_index=True)
	print(n, '\r', end = '')
data2 = pd.read_csv('D:\\Work\\Acad\\BTP\\data\\testDigitalPersona\\feature.csv', header=None)
data2['bw'] = data
data2.to_csv('D:\\Work\\Acad\\BTP\\data\\testDigitalPersona\\feature3.csv', header=False)