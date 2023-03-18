#######################
#      Problem 2      #
#######################
import cv2 as cv
import numpy as np

prevCircle = None
dist = lambda x1,y1,x2,y2: (x1-x2)**2 + (y1-y2)**2
x = []
y = []

vid = cv.VideoCapture('Videos/ball.mov')
while True:
    # Read in video
    isTrue, frame = vid.read()
    if not isTrue:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (13,13), 0)
    circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1.18, 150, param1=65,
                              param2=15, minRadius=9, maxRadius=13)

    np.seterr(over='ignore')
    if circles is not None:
        circles = np.uint16(np.around(circles))
        currCircle = None
        for i in circles[0,:]:
            if currCircle is None:
                currCircle = i
            if prevCircle is not None:
                if dist(currCircle[0],currCircle[1],prevCircle[0],prevCircle[1]) <= dist(i[0],i[1],prevCircle[0],prevCircle[1]):
                    currCircle = i
        cv.circle(frame, (currCircle[0], currCircle[1]),1, (255, 255, 255), 1)
        cv.circle(frame, (currCircle[0], currCircle[1]),11, (255, 255, 255), 1)
        x.append(currCircle[0])
        y.append(currCircle[1])
        prevCircle = currCircle

    if len(x) > 0:
        for i in range(0,len(x) - 1):
            cv.line(frame, (x[i],y[i]), (x[i+1],y[i+1]), (255, 255, 255), 2)
            cv.circle(frame, (x[i],y[i]), 10, (0, 0, 255), 1)

    cv.imshow('Frame', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

vid.release()
cv.destroyAllWindows()

#######################
#      Problem 3      #
#######################

img = cv.imread("Photos/train_track.jpg")
resized_img = cv.resize(img, (0, 0), fx=0.3, fy=0.3)

pt1 = (385, 370)
pt2 = (520, 370)
pt3 = (200, 590)
pt4 = (710, 590)

cv.circle(resized_img, pt1, 5, (0,0,255), -1)
cv.circle(resized_img, pt2, 5, (0,0,255), -1)
cv.circle(resized_img, pt3, 5, (0,0,255), -1)
cv.circle(resized_img, pt4, 5, (0,0,255), -1)

org_img = np.float32([pt1, pt2, pt3, pt4])
transformed_img = np.float32([[0,0],[400,0],[0,400], [400,400]])
homography_matrix = cv.getPerspectiveTransform(org_img, transformed_img)

top_view = cv.warpPerspective(resized_img, homography_matrix, (400,300))
gray = cv.cvtColor(top_view, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (17,17),0)
edges = cv.Canny(blur, 150, 220)


lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=15, maxLineGap=5)

X1 = []
X2 = []
for points in lines:
    # print(points)
    x1,y1,x2,y2 = points[0]
    if 0 <= abs(x2-x1) <= 2:
        cv.line(top_view, (x1, 0), (x2, 300), (0, 0, 255), 1)
        if x1 < 200:
            X1.append(x1)
        else:
            X2.append(x1)

X1_mean = np.mean(X1)
X2_mean = np.mean(X2)
print("The average distance between the outer part of the rails:")
print(abs(X1_mean - X2_mean))

cv.imshow('Top View',top_view)
cv.imshow('Canny', edges)
cv.imshow('Original image', resized_img)

cv.waitKey(0)
cv.destroyAllWindows()

#######################
#      Problem 4      #
#######################

img = cv.imread('Photos/hotairbaloon.jpg')
resized_img = cv.resize(img, (0, 0), fx=0.2, fy=0.2)

gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (15, 15), 0)
canny = cv.Canny(blur, 80, 170, 5, L2gradient=True)
dilated = cv.dilate(canny, (5, 5), iterations=2)
erode = cv.erode(dilated, (7, 7), iterations=2)

contours, hierarchy = cv.findContours(erode.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)

colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0),
           (255, 0, 128), (128, 255, 0), (128, 0, 255), (0, 255, 128), (0, 128, 255), (255, 128, 128),(128, 255, 128),
           (128, 128, 255), (255, 255, 255)]

for (i, j) in enumerate(sorted_contours):
    M = cv.moments(j)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    # cv.drawContours(resized_img, sorted_contours, i, colours[i], 2)
    cv.putText(resized_img, text=str(i+1), org=(cx, cy),
               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colours[i],
               thickness=2, lineType=cv.LINE_AA)

print("")
print("Number of balloons detected : ", len(contours))
cv.imshow('Labeled Image', resized_img)

cv.waitKey(0)
cv.destroyAllWindows()
