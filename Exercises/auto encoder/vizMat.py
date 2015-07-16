import numpy as np
import cv2
dataMat = np.loadtxt('/home/jason/Desktop/ufldl/auto encoder/W1MAT.mat' )
# size of dataMat is 25*64 i.e. there are 25 8 x 8 patches learnt
# let totalImages to pick  = 25

print dataMat.shape
visMat = None
hMat = None
for i in range(25):
	img1 = dataMat[i, :]
	img1 = np.reshape(img1, (8, 8))
	if(hMat == None):
		hMat = img1
	else:
		hMat = np.hstack((hMat, img1))
	print hMat.shape
	if((i+1)%5==0):
		if(visMat == None):
			visMat = hMat
		else:
			visMat = np.vstack((visMat, hMat))
		hMat = None
		print '---'
visMat = visMat - np.mean(visMat)
cv2.imwrite("visMat.jpg", visMat*255);

# import numpy as np
# import cv2
# dataMat = np.loadtxt('/home/jason/Desktop/ufldl/auto encoder/W1MAT.mat' )
# # size of dataMat is 25*64 i.e. there are 25 8 x 8 patches learnt
# # let totalImages to pick  = 25
# totalImages = 25
# visMat = None
# width = 5 # width == height
# for i in range(5):
# 	for j in range(5):
# 		ind = (i*width)+j
# 		print "index: ", ind , "subarray: ", i, j
# 		img1 = dataMat[ind, :]
# 		img1 = np.reshape(img1, (8,8))
# 		cv2.imwrite("./images/"+str(ind)+".jpg", img1*255)
# 		if(ind ==0):
# 			visMat = img1
# 		visMat = np.vstack((visMat, img1))

# cv2.imshow("visMat", visMat)
# cv2.waitKey(0)