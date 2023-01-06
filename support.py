import cv2
import numpy as np

def get_recMax(contour):
    rect = []
    for con in contour:
        area = cv2.contourArea(con)
        if area > 10:
            peri = cv2.arcLength(con,True)
            approx = cv2.approxPolyDP(con,0.02*peri,True)
            if len(approx) >=1:
                rect.append(con)
    rect = sorted(rect,key=cv2.contourArea,reverse=True)
    return rect

def get_corn(contour):
    peri = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.02*peri,True)
    approx = approx.tolist()
    return_list = []
    for i in range(len(approx)):
        return_list.append(approx[i][0])
    return return_list

def rearrage_point(rect_cont):
    length = rect_cont.shape[0]
    points = rect_cont.reshape((length,2))
    new_points = np.zeros((4,1), np.int32)
    add = points.sum(1)
    new_points[0] = np.argmin(add)
    new_points[3] = np.argmax(add)
    diff = np.diff(points,axis=1)
    new_points[1] = np.argmin(diff)
    new_points[2] = np.argmax(diff)
    return new_points

def split(img,numr,numc):
    rows = np.vsplit(img,numr)
    list_img = []
    list_temp = []
    for r in rows:
        cols = np.hsplit(r,numc)
        for c in cols:
            c1 = cv2.Canny(c,70,200)
            list_img.append(c1)
            list_temp.append(c)
    return list_img

def get_pointFrame(list_contour):
    temp_list = []
    for i in range(len(list_contour)):
        temp = list_contour[i][0].sum(1)
        temp_list.append(temp)
    print(temp_list)
    topLeft = list_contour[np.argmin(temp_list)]
    downRight = list_contour[np.argmax(temp_list)]
    pt1 = rearrage_point(topLeft)
    pt2 = rearrage_point(downRight)
    return pt1,pt2

def total_point(point):
    return np.sum(point)



