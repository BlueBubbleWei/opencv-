import cv2
import numpy as np
import sys
imgpath=sys.argv[1]
img=cv2.imread(imgpath)
# img=cv2.imread('chess.png')
alg=sys.argv[2]
# alg='SIFT'
def fd(algorithm):
    if algorithm == 'SIFT':
        return cv2.xfeatures2d.SIFT_create()
    if algorithm == 'SURF':
        return cv3.xfeatures2d.SIFT_create(float(sys.argv[3]) if len(sys.argv) ==4 else 4000)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

fd_alg=fd(alg)
keypoints,descriptor=fd_alg.detectAndCompute(gray,None)
img=cv2.drawKeypoints(image=img,outImage=img,keypoints=keypoints,flags=4,color=(51,163,236))
cv2.imshow('keypoints',img)
while(True):
    if cv2.waitKey(3) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()
