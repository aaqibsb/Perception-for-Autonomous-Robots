import glob
import cv2 as cv
import numpy as np

# Initializing empty lists for image and world points
world_pts = []
img_pts = []
K = None

# Setting up world coordinates with the given size of the checkboard squares
world_coord = np.zeros((6*9, 3), np.float32)
world_coord[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
world_coord = 21.5*world_coord
iterations = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Reading in all images at once
images = glob.glob("Calibration_Imgs/*.jpg")
k = 0
for image in images:
    k += 1
    image_points = []
    world_points = []

    # Resizing and Changing to gray scale image
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Finding chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (9, 6))

    # If corners found successfully
    if ret:
        # Add world points to the list
        world_pts.append(world_coord)
        world_points.append(world_coord)

        # Refine the found corners
        better_corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), iterations)

        # Add better found corners to the list
        img_pts.append(better_corners)
        image_points.append(better_corners)

        # Draw the refined found corners
        cv.drawChessboardCorners(img, (9, 6), better_corners, ret)

        # Find Intrinsic, Distortion, Rotation and Translation matrices
        ret, K, distort, R, T = cv.calibrateCamera(world_pts, img_pts, gray.shape[::-1], None, None)
        ret_n, K_n, distort_n, R_n, T_n = cv.calibrateCamera(world_points, image_points, gray.shape[::-1], None, None)
        print('Intrinsic Matrix (K) - image {}:'.format(k))
        print(K)

        # Finding re-projection error for all images and each image
        mean_error = 0
        for i in range(len(world_pts)):
            new_img_pts, _ = cv.projectPoints(world_pts[i], R[i], T[i], K, distort)
            error = cv.norm(img_pts[i], new_img_pts, cv.NORM_L2) / len(new_img_pts)
            mean_error += error

        total_error = mean_error / len(world_pts)
        print('Total re-projection error till image {}: '.format(k), np.round(total_error, 4))

        mean_error_n = 0
        for i in range(len(world_points)):
            new_img_points, _ = cv.projectPoints(world_points[i], R_n[i], T_n[i], K_n, distort_n)
            error_n = cv.norm(image_points[i], new_img_points, cv.NORM_L2) / len(new_img_points)
            mean_error_n += error_n

        total_error_n = mean_error_n / len(world_points)
        print('Re-projection error for image {}: '.format(k), np.round(total_error_n, 4))
        print()

        cv.imshow('Found Corners', img)
        cv.waitKey(0)

print('Final Intrinsic Matrix (K):')
print(K)
