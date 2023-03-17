import cv2 as cv
import numpy as np
import random


""" Extracting points from features """
def extract_points(match, kp1, kp2):
    pairs = []

    for i in match:
        img1_idx = i.queryIdx
        img2_idx = i.trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        pairs.append([(x1, y1),(x2, y2)])

    return pairs


""" Extracting feature matches """
def features(img1, img2):
    sift = cv.SIFT_create()
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    keyp1, des1 = sift.detectAndCompute(img1, None)
    keyp2, des2 = sift.detectAndCompute(img2, None)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:100]

    # output_image = cv.drawMatches(img1,keyp1,img2,keyp1,matches,None,flags=2)
    # cv.imshow('Matched Features',output_image)

    matched_points = extract_points(matches, keyp1, keyp2)

    return matched_points


""" Calculating homography matrix """
def homography(s, d):
    (xs1,ys1),(xs2,ys2),(xs3,ys3),(xs4,ys4) = s
    (xd1,yd1),(xd2,yd2),(xd3,yd3),(xd4,yd4) = d

    A = np.array([[  xd1,   yd1, 1,    0,   0, 0,  -xs1*xd1,  -xs1*yd1, -xs1],
                  [    0,     0, 0,  xd1, yd1, 1,  -ys1*xd1,  -ys1*yd1, -ys1],
                  [  xd2,   yd2, 1,    0,   0, 0,  -xs2*xd2,  -xs2*yd2, -xs2],
                  [    0,     0, 0,  xd2, yd2, 1,  -ys2*xd2,  -ys2*yd2, -ys2],
                  [  xd3,   yd3, 1,    0,   0, 0,  -xs3*xd3,  -xs3*yd3, -xs3],
                  [    0,     0, 0,  xd3, yd3, 1,  -ys3*xd3,  -ys3*yd3, -ys3],
                  [  xd4,   yd4, 1,    0,   0, 0,  -xs4*xd4,  -xs4*yd4, -xs4],
                  [    0,     0, 0,  xd4, yd4, 1,  -ys4*xd4,  -ys4*yd4, -ys4]])

    H = A.T.dot(A)
    eigenvalues, eigenvectors = np.linalg.eig(H)
    smallest_eigenvalue_index = np.argmin(eigenvalues)
    h = eigenvectors[:, smallest_eigenvalue_index]
    h = h.reshape((3, 3))
    h = h/h[-1][-1]
    return h


""" Error Function """
def error(data, homo_matrix):
    (xs,ys),(xd,yd) = data

    Xs = np.array([[xs],
                   [ys],
                   [1]])

    Xd = np.array([[xd],
                   [yd],
                   [1]])

    X_pred = homo_matrix @ Xd
    X_pred = X_pred/X_pred[-1][-1]

    computed_error = np.linalg.norm(Xs - X_pred)
    return computed_error


""" RANSAC function """
def ransac(data, k, t, d):
    best_h = None

    for i in range(k):
        sample_points = random.sample(data, 4)

        img1_random = [item[0] for item in sample_points]
        img2_random = [item[1] for item in sample_points]

        hyp_h = homography(img1_random, img2_random)
        inliers = []

        for point in data:
            if point not in sample_points:
                if error(point,hyp_h) < t:
                    inliers.append(point)

        if len(inliers) > d:
            best_h = hyp_h
    return best_h


# Reading images
frame1 = cv.imread('Photos/image_1.jpg')
frame2 = cv.imread('Photos/image_2.jpg')
frame3 = cv.imread('Photos/image_3.jpg')
frame4 = cv.imread('Photos/image_4.jpg')

# Resizing images
frame1 = cv.resize(frame1, (0, 0), fx=0.178, fy=0.178)
frame2 = cv.resize(frame2, (0, 0), fx=0.178, fy=0.178)
frame3 = cv.resize(frame3, (0, 0), fx=0.178, fy=0.178)
frame4 = cv.resize(frame4, (0, 0), fx=0.178, fy=0.178)

# Calculating homography between image1 and image2
matched_points1 = features(frame1,frame2)
H12 = ransac(matched_points1, 1000, 10, 90)

# Calculating homography between image2 and image3
matched_points2 = features(frame2,frame3)
H23 = ransac(matched_points2, 1000, 10, 90)

# Calculating homography between image3 and image4
matched_points3 = features(frame3,frame4)
H34 = ransac(matched_points3, 1000, 10, 90)


height_1, width_1 = frame1.shape[:2]
height_2, width_2 = frame2.shape[:2]
height_3, width_3 = frame3.shape[:2]
height_4, width_4 = frame4.shape[:2]

# Stitching image1 and image2 
image12 = (cv.warpPerspective(frame2, H12, (width_1+width_2, height_1)))
stitch = image12.copy()
stitch[0:frame1.shape[0],0:frame1.shape[1]] = frame1
cv.imshow('IMAGE12', stitch)

# Stitching image2 and image3 
image23 = (cv.warpPerspective(frame3, H23, (width_2+width_3, height_2)))
stitch = image23.copy()
stitch[0:frame2.shape[0],0:frame2.shape[1]] = frame2
cv.imshow('IMAGE23', stitch)

# Stitching image3 and image4 
image34 = (cv.warpPerspective(frame4, H34, (width_3+width_4, height_3)))
stitch = image34.copy()
stitch[0:frame3.shape[0],0:frame3.shape[1]] = frame3
cv.imshow('IMAGE34', stitch)

cv.waitKey(0)





