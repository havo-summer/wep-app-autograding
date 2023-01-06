import cv2
import numpy as np

cap = cv2.VideoCapture(0)
img1 = cv2.imread("phieu.png")
h1,w1 = img1.shape[:2]

cv2.namedWindow("camera",cv2.WINDOW_NORMAL)
cv2.resizeWindow("camera",640,800)


color = (0,0,255)
while True:
    success,img = cap.read()
    if success:
        h,w = img.shape[:2]
        x = w//2
        x1 = int(x - x//2)
        x2 = int(x + x//2)
        w_sub = x2 - x1
        y1 = 50
        y2 = y1+(h1*w_sub)//w1
        
        cv2.rectangle(img,(x1,y1),(x2,y2),color,1)
        pt1 = [x1,y1]
        pt2 = [x2,y1]
        pt3 = [x1,y2]
        pt4 = [x2,y2]

        pts1 = np.float32([pt1,pt2,pt3,pt4])
        pts2 = np.float32([[0,0],[w1,0],[0,h1],[w1,h1]])

        matrix = cv2.getPerspectiveTransform(pts1,pts2)
        img_test = cv2.warpPerspective(img,matrix,(w1,h1))

        orb = cv2.ORB_create(nfeatures=1000)

        kp1,des1 = orb.detectAndCompute(img_test,None)
        kp2,des2 = orb.detectAndCompute(img1,None)

        imgKp1 = cv2.drawKeypoints(img_test,kp1,None)
        imgKp2 = cv2.drawKeypoints(img1,kp2,None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des2,des1,k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.7 * n.distance:
                good.append([m])
        print(len(good))
        img3 = cv2.drawMatchesKnn(img_test,kp1,img1,kp2,good,None,flags=2)


        cv2.imshow("t1",img3)
        cv2.imshow("camera",img_test)
        if cv2.waitKey(1) and 0xff == 'q':
            break
    else:
        break