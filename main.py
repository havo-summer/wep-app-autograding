import cv2
import support
import numpy as np

img = cv2.imread("phieu5.jpg")
width = 550
w,h = img.shape[1],img.shape[0]
height = int((width*h)/w)
img = cv2.resize(img,(width,height))
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


img_canny = cv2.Canny(img_gray,400,550)
img2 = img.copy()

contour,_ = cv2.findContours(img_canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
list_contour = []
for con in contour:
    area = cv2.contourArea(con)
    if area > 100 :
        peri = cv2.arcLength(con,True)
        approx = cv2.approxPolyDP(con,peri*0.02,True)
        # print(len(approx))
        if len(approx)  == 4 or len(approx) ==5:
            list_contour.append(con)
cv2.drawContours(img2,list_contour,-1,(0,255,0),2)

for i in range(len(list_contour)):
    corn = support.get_corn(list_contour[i])
    corn = np.array(corn)
    x = support.rearrage_point(corn)
    new_points = [[],[],[],[]]
    new_points[0] = corn[x[0][0]]
    new_points[1] = corn[x[1][0]]
    new_points[2] = corn[x[2][0]]
    new_points[3] = corn[x[3][0]]
    list_contour[i] = new_points

list_contour = sorted(list_contour,key = support.total_point)
print(len(list_contour))
list_contour = np.array(list_contour)
print(list_contour)
cv2.waitKey(0)
x_index = support.rearrage_point(list_contour.sum(1))
print(x_index)
print(list_contour.sum(1))
x1 = list_contour[x_index[0][0]][3]
x2 = list_contour[x_index[1][0]][2]
x3 = list_contour[x_index[2][0]][1]
x4 = list_contour[x_index[3][0]][0]
print(x1,x2,x3,x4)


pts1 = np.float32([x1,x2,x3,x4])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgwrap = cv2.warpPerspective(img_gray,matrix,(width,height))
img_BD = imgwrap.copy()

wrap_canny = cv2.Canny(imgwrap,300,550)
cont,_ = cv2.findContours(wrap_canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
ListCon = []
for c in cont:
    area = cv2.contourArea(c)
    if area > 0:
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,peri*0.02,True)
        if len(approx) == 4 or len(approx) == 5:
            ListCon.append(c)
cv2.drawContours(imgwrap,ListCon,-1,(0,255,0),2)
for i in range(len(ListCon)):
    corn = support.get_corn(ListCon[i])
    print(corn)
    corns = np.array(corn)
    x = support.rearrage_point(corns)
    new_points = [[],[],[],[]]
    new_points[0] = corn[x[0][0]]
    new_points[1] = corn[x[1][0]]
    new_points[2] = corn[x[2][0]]
    new_points[3] = corn[x[3][0]]
    ListCon[i] = new_points
ListCon = sorted(ListCon,key=support.total_point)
print(ListCon)
d = ListCon[1][0][0] - ListCon[0][0][0]
dv1 = int(d/6)
dv3 = int(d/2)
dv10 = dv1 * 10
##################################################################
#Lay so bao danh
wBD = 480
hBD = 700
numRow = 10
numCow = 6


x1 = [ListCon[0][0][0] - d,ListCon[0][0][1]]
x4 = ListCon[2][0]
x2 = [x4[0],x1[1]]
x3 = [x1[0],x4[1]]
print(x1,x2,x3,x4)
pts1 = np.float32([x1,x2,x3,x4])
pts2 = np.float32([[0,0],[wBD,0],[0,hBD],[wBD,hBD]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgBD = cv2.warpPerspective(img_BD,matrix,(wBD,hBD))
# cv2.imshow("SBD",imgBD)

sub_imgs = support.split(imgBD,10,6)
r,c = 0,0
myList = np.zeros((10,6))
for i in sub_imgs: 
    myList[r][c] = cv2.countNonZero(i)
    c += 1
    if c == 6:
        c = 0
        r+= 1 
myList = myList.T
SBD = []
for i in range(0,6):
    arr = myList[i]
    index = np.where(arr == np.amax(arr))
    SBD.append(index[0][0])
SBD = "".join(str(s) for s in SBD)
print(SBD)
##################################################################
# Lay ma de
wD = 360
hD = 500
x1 = ListCon[0][3]
x1[0] = x1[0] +4
x4 = [x1[0] + dv3,ListCon[2][0][1]]
x2 = [x4[0],x1[1]]
x3 = [x1[0],x4[1]]
pts1 = np.float32([x1,x2,x3,x4])
pts2 = np.float32([[0,0],[wD,0],[0,hD],[wD,hD]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgD = cv2.warpPerspective(img_BD,matrix,(wD,hD))

subMD = support.split(imgD,10,3)
r,c = 0,0
myList = np.zeros((10,3))
for i in subMD: 
    myList[r][c] = cv2.countNonZero(i)
    c += 1
    if c == 3:
        c = 0
        r+= 1 
myList = myList.T
MD = []
for i in range(0,3):
    arr = myList[i]
    index = np.where(arr == np.amax(arr))
    MD.append(index[0][0])
MD = "".join(str(s) for s in MD)
print(MD)
#cv2.imshow("MA DE",imgD)
##################################################################
# cham diem
score = []
wS,hS = 200,600
###############################
#frame 1 | 1->10


getX = ListCon[1][3]
x1 = [getX[0] +10,getX[1]]
x4 = [x1[0] + dv1*4,ListCon[2][0][1]-10]
x2 = [x4[0],x1[1]]
x3 = [x1[0],x4[1]]
# print(x1,x2,x3,x4)
# cv2.rectangle(img_BD,x1,x4,(0,255,0))

pts1 = np.float32([x1,x2,x3,x4])
pts2 = np.float32([[0,0],[wS,0],[0,hS],[wS,hS]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgS1 = cv2.warpPerspective(img_BD,matrix,(wS,hS))

subMS1= support.split(imgS1,10,4)
r,c = 0,0
myList = np.zeros((10,4))
for i in subMS1: 
    myList[r][c] = cv2.countNonZero(i)
    c += 1
    if c == 4:
        c = 0
        r+= 1 
j = 0

for i in range(0,10):
    arr = myList[i]
    index = np.where(arr == np.amax(arr))
    score.append(index[0][0])


#cv2.imshow("1->10",imgS1)

###############################
# frame 2 11 -> 20
getX1 = ListCon[2][2]
getX2 = ListCon[4][0]
x1 = [getX1[0]-dv1*5,getX1[1]+10]
x4 = [getX1[0]-dv1,getX2[1]-10]
x2 = [x4[0],x1[1]]
x3 = [x1[0],x4[1]]
print(x1,x2,x3,x4)
pts1 = np.float32([x1,x2,x3,x4])
pts2 = np.float32([[0,0],[wS,0],[0,hS],[wS,hS]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgS2 = cv2.warpPerspective(img_BD,matrix,(wS,hS))
# cv2.imshow("11->20",imgS2)

subMS2= support.split(imgS2,10,4)
r,c = 0,0
myList = np.zeros((10,4))
for i in subMS2: 
    myList[r][c] = cv2.countNonZero(i)
    c += 1
    if c == 4:
        c = 0
        r+= 1 
j = 0

for i in range(0,10):
    arr = myList[i]
    index = np.where(arr == np.amax(arr))
    score.append(index[0][0])
###############################
# frame 3 21 -> 30
getX1 = ListCon[2][3]
getX2 = ListCon[4][0]

x1 = [getX1[0]+10,getX1[1]+10]
x4 = [x1[0] + dv1*4,getX2[1]-10]
x2 = [x4[0],x1[1]]
x3 = [x1[0],x4[1]]
print(x1,x2,x3,x4)
pts1 = np.float32([x1,x2,x3,x4])
pts2 = np.float32([[0,0],[wS,0],[0,hS],[wS,hS]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgS3 = cv2.warpPerspective(img_BD,matrix,(wS,hS))
#cv2.imshow("21 -> 30",imgS3)

subMS3= support.split(imgS3,10,4)
r,c = 0,0
myList = np.zeros((10,4))
for i in subMS3: 
    myList[r][c] = cv2.countNonZero(i)
    c += 1
    if c == 4:
        c = 0
        r+= 1 
j = 0

for i in range(0,10):
    arr = myList[i]
    index = np.where(arr == np.amax(arr))
    score.append(index[0][0])

###############################
# frame 3 21 -> 30

getX1 = ListCon[3][3]
getX2 = ListCon[4][0]

x1 = [getX1[0]+10,getX1[1]+10]
x4 = [x1[0] + dv1 * 4, getX2[1]-10]
x2 = [x4[0],x1[1]]
x3 = [x1[0],x4[1]]
print(x1,x2,x3,x4)
pts1 = np.float32([x1,x2,x3,x4])
pts2 = np.float32([[0,0],[wS,0],[0,hS],[wS,hS]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgS4 = cv2.warpPerspective(img_BD,matrix,(wS,hS))
cv2.imshow("31 -> 40",imgS4)

subMS4= support.split(imgS4,10,4)
r,c = 0,0
myList = np.zeros((10,4))
for i in subMS4: 
    myList[r][c] = cv2.countNonZero(i)
    c += 1
    if c == 4:
        c = 0
        r+= 1 
j = 0

for i in range(0,10):
    arr = myList[i]
    index = np.where(arr == np.amax(arr))
    score.append(index[0][0])

print(score)
#cv2.imshow("test1",imgwrap)
cv2.imshow("test",img_BD)
cv2.waitKey(0)
