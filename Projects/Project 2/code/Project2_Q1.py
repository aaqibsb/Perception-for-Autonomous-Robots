import cv2 as cv
from sympy import *
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import itertools

X = []
Y = []
Z = []

X1 = []
Y1 = []
Z1 = []
Hough = {}
theta = np.arange(0,180)

""" Hough Transform Function """
def hough_transform(coordinates):
    Hough.clear()
    for it in range(len(coordinates)):
        for t in theta:
            d = (int(coordinates[it,0,0]*np.cos(t*np.pi/180) + coordinates[it,0,1]*np.sin(t*np.pi/180)))/10
            if (d,t) in Hough:
                Hough[d,t] += 1
            else:
                Hough[d,t] = 0
    return Hough


""" Checking for corners """
def check_perpendicular(h_dict):
    perp_points = []
    R_t = list(h_dict.keys())
    i = 0
    for first in R_t:
        R1, t1 = first
        for second in R_t[i:]:
            R2, t2 = second
            if 83 < abs(t2 - t1) < 97:
                perp_points.append([(R1,t1), (R2, t2)])
        i += 1

    return perp_points


""" Getting corner points """
def get_coordinates(ang):
    x, y = symbols('x y')
    points = []

    for i in range(len(ang)):
        eq1 = Eq(x*np.cos(ang[i][0][1]*np.pi/180) + y*np.sin(ang[i][0][1]*np.pi/180) - ang[i][0][0]*10, 0)
        eq2 = Eq(x*np.cos(ang[i][1][1]*np.pi/180) + y*np.sin(ang[i][1][1]*np.pi/180) - ang[i][1][0]*10, 0)
        res = solve((eq1,eq2), x,y)

        if res:
            X = int(res[x].evalf())
            Y = int(res[y].evalf())
            points.append((X,Y))
            cv.circle(frame, (X, Y), 2, (0, 0, 255), -1)
        else:
            continue

    return points


""" Calculating Manhattan distance """
def Manhattan(tup1, tup2):
    return abs(tup1[0] - tup2[0]) + abs(tup1[1] - tup2[1])


""" Finding the final four corners for each frame """
def four_corners(points):
    # print(points)
    main_list = [sorted(it) for it in itertools.product(points, repeat=2) if Manhattan(*it) < 14.6]

    final_dict = {el: {el} for el in points}
    for tup1, tup2 in main_list:
        final_dict[tup1] |= final_dict[tup2]
        final_dict[tup2] = final_dict[tup1]

    res = [[*next(val)] for key, val in itertools.groupby(sorted(final_dict.values(), key=id), id)]

    X1 = int(sum([x1[0] for x1 in res[0]])/len(res[0]))
    Y1 = int(sum([y1[1] for y1 in res[0]])/len(res[0]))
    X2 = int(sum([x2[0] for x2 in res[1]])/len(res[1]))
    Y2 = int(sum([y2[1] for y2 in res[1]])/len(res[1]))
    X3 = int(sum([x3[0] for x3 in res[2]])/len(res[2]))
    Y3 = int(sum([y3[1] for y3 in res[2]])/len(res[2]))
    X4 = int(sum([x4[0] for x4 in res[3]])/len(res[3]))
    Y4 = int(sum([y4[1] for y4 in res[3]])/len(res[3]))

    cv.circle(frame,(X1,Y1), 5, (255, 255, 0), -1)
    cv.circle(frame,(X2,Y2), 5, (255, 255, 0), -1)
    cv.circle(frame,(X3,Y3), 5, (255, 255, 0), -1)
    cv.circle(frame,(X4,Y4), 5, (255, 255, 0), -1)

    c_points = [(X1,Y1),(X2,Y2),(X3,Y3),(X4,Y4)]
    return c_points


""" Calculating homography matrix """
def homography(f_corners):
    sort_corners = sorted(f_corners, key=lambda x: x[0])
    (x1,y1),(x2,y2),(x3,y3),(x4,y4) = sort_corners

    A = np.array([[    0,    0,  1,    0,    0,   0,     -x1*0,    -x1*0,  -x1],
                  [    0,    0,  0,    0,    0,   1,     -y1*0,    -y1*0,  -y1],
                  [ 21.6,    0,  1,    0,    0,   0,  -x2*21.6,    -x2*0,  -x2],
                  [    0,    0,  0, 21.6,    0,   1,  -y2*21.6,    -y2*0,  -y2],
                  [    0, 27.9,  1,    0,    0,   0,     -x3*0,  -x3*27.9, -x3],
                  [    0,    0,  0,    0, 27.9,   1,     -y3*0,  -y3*27.9, -y3],
                  [ 21.6, 27.9,  1,    0,    0,   0,  -x4*21.6,  -x4*27.9, -x4],
                  [    0,    0,  0,   22, 27.9,   1,  -y4*21.6,  -y4*27.9, -y4]])
    # pprint(A)

    H = A.T @ A
    eigenvalues, eigenvectors = np.linalg.eig(H)
    smallest_eigenvalue_index = np.argmin(eigenvalues)
    h = eigenvectors[:, smallest_eigenvalue_index]

    h = h.reshape((3,3))
    h = h/h[-1][-1]

    K = np.array([[1382.58398,          0, 945.743164],
                  [         0, 1382.58398,  527.04834],
                  [         0,          0,          1]])

    B = np.linalg.inv(K) @ h

    B1 = B[:, 0]
    L1 = np.linalg.norm(B1)

    B2 = B[:, 1]
    L2 = np.linalg.norm(B2)

    L = (L1 + L2)/2

    Rot_t = B/L
    R1 = Rot_t[:, 0]
    R2 = Rot_t[:, 1]
    t = Rot_t[:, 2]
    R3 = np.cross(R1,R2, axis=0)

    R = np.array([R1, R2, R3]).T

    R = Rotation.from_matrix(R)
    euler_angles = R.as_euler('zyx', degrees=True)

    X.append(euler_angles[2])
    Y.append(euler_angles[1])
    Z.append(euler_angles[0])

    X1.append(t[0])
    Y1.append(t[1])
    Z1.append(t[2])


""" Reading video """
vid = cv.VideoCapture('Videos/project2.avi')
i = 0
while True:
    i += 1

    # Read in video
    isTrue, frame = vid.read()

    # Converting to gray-scale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Applying Gaussian Blur
    gauss = cv.GaussianBlur(gray, (11, 11), 0)

    # Applying Canny edge detector
    edge = cv.Canny(gauss, 130, 160)

    # Finding coordinates of edge points
    coord = cv.findNonZero(edge)

    # Computing hough transform for each frame
    single_hough = hough_transform(coord)
    reverse_hough = dict(sorted(single_hough.items(), key=lambda x:x[1], reverse=True))
    short_hough = dict(itertools.islice(reverse_hough.items(), 8))

    # Finding intersection points
    perp_angles = check_perpendicular(short_hough)

    # Getting corner coordinates
    group_corner_points = get_coordinates(perp_angles)

    # Clustering adjacent points
    final_corners = four_corners(group_corner_points)

    # Computing homography matrix
    homography(final_corners)

    cv.imshow('Frame', frame)
    print("Processing Frame:", i)
    if i == 147:
        break

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

vid.release()
cv.destroyAllWindows()


""" Plotting after entire video finishes playing """
fig, axs = plt.subplots(1, 2, figsize=(10, 10))
iterations = np.arange(1,148,1)
axs[0].plot(iterations, X, label='Roll')
axs[0].plot(iterations, Y, label='Pitch')
axs[0].plot(iterations, Z, label='Yaw')
axs[0].set_xlabel('Frames')
axs[0].set_ylabel('Angles')
axs[0].set_title('RPY')
axs[0].legend()

axs[1].plot(iterations, X1, label='X')
axs[1].plot(iterations, Y1, label='Y')
axs[1].plot(iterations, Z1, label='Z')
axs[1].set_xlabel('Frames')
axs[1].set_ylabel('Distance')
axs[1].set_title('Translations')
axs[1].legend()

plt.show()

