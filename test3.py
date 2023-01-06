import cv2
import numpy as np

circles = np.zeros((4,2),int)
count = 0

def mousePoint(event, x, y, flags, params):
    global count
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        circles[count] = x,y
        count = count + 1
        print(count)

img = cv2.imread("phieu5.jpg")
img1 = cv2.imread("phieu.png")

width1 = 400
w1,h1 = img1.shape[1],img.shape[0]
height1 = int((width1*h1)/w1)
img1 = cv2.resize(img1,(width1,height1))

width = 550
w,h = img.shape[1],img.shape[0]
height = int((width*h)/w)
img = cv2.resize(img,(width,height))

circles[0] = [23,39]
circles[1] = [491,12]
circles[2] = [33,719]
circles[3] = [533,726]

width, height = width1,height1
pts1 = np.float32([circles[0],circles[1],circles[2],circles[3]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
output = cv2.warpPerspective(img,matrix,(width,height))

img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img_blur = cv2.GaussianBlur(img_gray,(5,5),1)
img_canny = cv2.Canny(img_blur,100,450)

img1_gray = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
img1_blur = cv2.GaussianBlur(img1_gray,(5,5),1)
output = cv2.cvtColor(output,cv2.COLOR_RGB2GRAY)
output_blur = cv2.GaussianBlur(output,(7,7),1)

output_canny = cv2.Canny(output_blur,10,450)
img1_canny = cv2.Canny(img1_blur,10,450)

orb = cv2.ORB_create(nfeatures=1000)

kp1,des1 = orb.detectAndCompute(img1_canny,None)
kp2,des2 = orb.detectAndCompute(output_canny,None)

imgKp1 = cv2.drawKeypoints(img1_canny,kp1,None)
imgKp2 = cv2.drawKeypoints(output_canny,kp2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des2,des1,k=2)

good = []
for m,n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])
print(len(good))
img3 = cv2.drawMatchesKnn(img1_canny,kp1,output_canny,kp2,good,None,flags=2)


cv2.imshow("canny",img3)
cv2.imshow("phieu",img1_canny)
cv2.imshow("cut img",output_canny)
cv2.imshow("orignal img",img_canny)

cv2.waitKey(0)