import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

cap = cv2.VideoCapture('lilo.mp4')
i = 0

while(cap.isOpened()):
    ret, currImage = cap.read()

    if currImage.size == 0:
        print('No frame')
        break

    if i == 0 :
        prevImage = currImage.copy()

    sift = cv2.xfeatures2d.SIFT_create()
    kpts1, desc1 = sift.detectAndCompute(currImage, None)
    kpts2, desc2 = sift.detectAndCompute(prevImage, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

##    ####################################
    matchesMask = [[0,0] for i in range(len(matches))]
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.55*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0),matchesMask = matchesMask, flags = 0)
    img3 = cv2.drawMatchesKnn(currImage, kpts1, prevImage, kpts2, matches,None, **draw_params)
    
    matches = np.asarray(matchesMask)
    print(len(matches))
##    ####################################


    ####################################
##    good = []
##    for m in matches :
##        if m[0].distance < 0.5*m[1].distance :
##            good.append(m)
##    matches = np.asarray(good)
##    draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0),matchesMask = good, flags = 0)
##    img3 = cv2.drawMatchesKnn(currImage, kpts1, prevImage, kpts2, good,None, **draw_params)
##    #img3 = cv2.drawMatchesKnn(currImage, kpts1, prevImage, kpts2,good,flags=2, outImg = None)
    ####################################
    

    if len(matches[:,0]) >= 4:
        src = np.float32([ kpts1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kpts2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)

        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        print(H)
    else:
        raise AssertionError("Can't find enough keypoints.")

    dst = cv2.warpPerspective(currImage,H,(currImage.shape[1] + currImage.shape[1], currImage.shape[0]))     	
    dst[0:currImage.shape[0], 0:currImage.shape[1]] = currImage
    
    cv2.imshow('resultant_stitched_panorama.jpg',dst)

    cv2.imshow('prev',prevImage)
    cv2.imshow('curr',currImage)
    cv2.imshow('match', img3)
    
    i+=1

    cv2.waitKey(0)
    
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

cap.release()
cv2.waitKey(0)
