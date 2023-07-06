import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


""" Extracting points from features """
def extract_points(match, kp1, kp2):
    pairs = []

    for i in match:
        img1_idx = i.queryIdx
        img2_idx = i.trainIdx

        (X1, Y1) = kp1[img1_idx].pt
        (X2, Y2) = kp2[img2_idx].pt

        pairs.append([(X1, Y1),(X2, Y2)])

    return pairs


""" Extracting feature matches """
def features(d_img1, d_img2):
    sift = cv.SIFT_create()
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

    d_img1 = cv.cvtColor(d_img1, cv.COLOR_BGR2GRAY)
    d_img2 = cv.cvtColor(d_img2, cv.COLOR_BGR2GRAY)

    keyp1, des1 = sift.detectAndCompute(d_img1, None)
    keyp2, des2 = sift.detectAndCompute(d_img2, None)

    matches = bf.match(des1, des2)
    matches_n = sorted(matches, key=lambda x: x.distance)[:100]

    # output_image = cv.drawMatches(img1,keyp1,img2,keyp1,matches_n,None,flags=2)
    # cv.imshow('Matched Features',output_image)

    matched_pts = extract_points(matches, keyp1, keyp2)

    return matched_pts


""" Calculating Fundamental matrix """
def fundamental_matrix(it, sample, s, d):
    A = np.zeros((8,9))
    for i in range(8):
        A[i][0] = s[sample[it]][0]*d[sample[it]][0]
        A[i][1] = s[sample[it]][1]*d[sample[it]][0]
        A[i][2] = d[sample[it]][0]
        A[i][3] = s[sample[it]][0]*d[sample[it]][1]
        A[i][4] = s[sample[it]][1]*d[sample[it]][1]
        A[i][5] = d[sample[it]][1]
        A[i][6] = s[sample[it]][0]
        A[i][7] = s[sample[it]][1]
        A[i][8] = 1
        it += 1

    _, _, V = np.linalg.svd(A)
    f = np.reshape(V[-1], (3, 3))

    u, s, v = np.linalg.svd(f)
    s[2] = 0

    F_mtx = u @ np.diag(s) @ v
    F_mtx = F_mtx / F_mtx[-1, -1]
    return F_mtx


""" RANSAC function """
def ransac(data, t, d):
    best_f = None
    it = 0
    i = 0
    k = int(np.log(1 - 0.99) / np.log(1 - (1 - 0.75) ** 8))

    np.random.seed(0)
    sample_index = np.random.choice(len(data), k*8)

    while i < k:
        img1_pts = [item[0] for item in data]
        img2_pts = [item[1] for item in data]

        hyp_f = fundamental_matrix(it, sample_index, img1_pts, img2_pts)

        ones = np.ones([len(matched_points), 1])
        _img1 = np.hstack([img1_pts, ones])
        _img2 = np.hstack([img2_pts, ones])
        inliers = []

        error = abs(np.diag(_img2 @ hyp_f @ _img1.T))

        for val in error:
            if val < t:
                inliers.append(val)

        if len(inliers) > d:
            best_f = hyp_f
            break

        it += 8
        i += 1

    return best_f


""" Triangulating Points """
def triangulate_points(rot, tr):
    Projection1 = np.dot(K_intrinsic, np.hstack((np.eye(3), np.zeros((3, 1)))))
    Projection2 = np.dot(K_intrinsic, np.hstack((rot, tr.reshape(-1, 1))))

    world_pts = cv.triangulatePoints(Projection1, Projection2, np.asarray(pts1[0:4]).T, np.asarray(pts2[0:4]).T)
    world_pts /= world_pts[3]

    it = 0
    for i in range(4):
        if world_pts[2][i] >= 0:
            it += 1
    return it


""" Decompose Essential Matrix """
def decompose_ess(ess):
    U, S, V = np.linalg.svd(ess)

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    r1 = U @ W @ V
    if np.linalg.det(r1) < 0:
        r1 = -r1

    r2 = U @ W.T @ V
    if np.linalg.det(r2) < 0:
        r2 = -r2

    t1 = U[:, 2]
    t2 = -U[:, 2]

    a = triangulate_points(r1, t1)
    b = triangulate_points(r1, t2)
    c = triangulate_points(r2, t1)
    d = triangulate_points(r2, t2)

    max_count = max(a, b, c, d)
    if a == max_count:
        return r1, t1
    elif b == max_count:
        return r1, t2
    elif c == max_count:
        return r2, t1
    else:
        return r2, t2


""" Calculate homography matrices """
def get_homos(im1, im2):
    height1, width1, _ = im1.shape
    height2, width2, _ = im2.shape
    _, Homo1, Homo2 = cv.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(width1, height1))
    print('Homography matrix for image 1:\n', Homo1, sep='')
    print()
    print('Homography matrix for image 2:\n', Homo2, sep='')
    print()

    return Homo1, Homo2


""" Draw epipolar lines """
def drawlines(_img1,_img2,lines,_pts1,_pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = _img1.shape
    _img1 = cv.cvtColor(_img1,cv.COLOR_GRAY2BGR)
    _img2 = cv.cvtColor(_img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,_pts1,_pts2):
        x_2,y_2 = map(int,tuple(pt1))
        x_3,y_3 = map(int,tuple(pt2))

        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1]])
        x_1,y_1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        _img1 = cv.line(_img1, (x0,y0), (x_1,y_1), color,1)
        _img1 = cv.circle(_img1,(x_2,y_2),5,color,-1)
        _img2 = cv.circle(_img2,(x_3,y_3),5,color,-1)
    return _img1,_img2


""" Calculating SSD """
def squared_diff(y_1, x_1, x_2):
    distance = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            distance += pow((int(gray_rect_img1[y_1 + i][x_1 + j]) - int(gray_rect_img2[y_1 + i][j + x_1 - x_2])),2)
    return distance


""" Display Epipolar lines """
def display_epipolar(im1, im2):
    img1_copy = im1.copy()
    img2_copy = im2.copy()
    img1 = cv.cvtColor(img1_copy, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2_copy, cv.COLOR_BGR2GRAY)

    first_lines = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    first_lines = first_lines.reshape(-1, 3)
    epi_img1, epi_img2 = drawlines(img1, img2, first_lines, pts1, pts2)

    second_lines = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    second_lines = second_lines.reshape(-1, 3)
    epi_img3, epi_img4 = drawlines(img2, img1, second_lines, pts2, pts1)

    img1_epi = cv.warpPerspective(epi_img1, H1, (w1, h1))
    img2_epi = cv.warpPerspective(epi_img3, H2, (w2, h2))
    plt.subplot(121), plt.imshow(img1_epi)
    plt.subplot(122), plt.imshow(img2_epi)
    plt.get_current_fig_manager().set_window_title('Epipolar Lines')
    plt.show()


""" Estimating disparity and depth """
def estimate_disparity_depth():
    Height, Width = gray_rect_img1.shape
    vmin = 55
    vmax = 142

    SSD = np.zeros(shape=(Width, Height))
    depth = np.zeros(shape=(Width, Height))
    disparity = None
    for y1 in np.arange(3, Height-3):
        for x1 in np.arange(3, Width-3):
            minimum_SD = float('inf')
            for x2 in np.arange(0, 30, 3):
                dist = squared_diff(y1, x1, x2)
                minimum_SD = min(minimum_SD, dist)
                if dist == minimum_SD:
                    disparity = x2
            SSD[x1][y1] = disparity
            if int(disparity) == 0:
                disparity = 0.00001
                depth[x1][y1] = (1733.74 * 536.62)/disparity
            else:
                depth[x1][y1] = (1733.74 * 536.62)/disparity

    _normalized_disparity = cv.normalize(SSD.T, None, vmin, vmax, cv.NORM_MINMAX, cv.CV_8U)
    _colormap_disparity = cv.applyColorMap(_normalized_disparity, cv.COLORMAP_JET)

    cv.imshow("Disparity Grayscale", _normalized_disparity)
    cv.imshow("Disparity Color Map", _colormap_disparity)

    cv.waitKey(0)
    cv.destroyAllWindows()

    _normalized_depth = cv.normalize(depth.T, None, vmin, vmax, cv.NORM_MINMAX, cv.CV_8U)
    _colormap_depth = cv.applyColorMap(_normalized_disparity, cv.COLORMAP_JET)

    cv.imshow("Depth Grayscale", _normalized_depth)
    cv.imshow("Depth Color Map", _colormap_depth)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    # Reading Images
    data_img1 = cv.imread("depth_files/artroom/im0.png")
    data_img2 = cv.imread("depth_files/artroom/im1.png")

    # Finding matching points
    matched_points = features(data_img1,data_img2)
    pts1 = np.array([t[0] for t in matched_points])
    pts2 = np.array([t[1] for t in matched_points])

    # Finding Fundamental Matrix using RANSAC
    F = ransac(matched_points, 1, 1500)
    print("Fundamental Matrix:\n",F,sep='')
    print()

    K_intrinsic = np.array([[1733.74, 0, 792.27],
                            [0, 1733.74, 541.89],
                            [0,       0,      1]])

    # Finding Essential Matrix
    E = K_intrinsic.T @ F @ K_intrinsic
    print("Essential Matrix:\n", E, sep='')
    print()

    # Decomposing Essential Matrix into R, T
    R,T = decompose_ess(E)
    print("Rotation Matrix:\n", R, sep='')
    print()
    print("Translation Matrix:\n", T, sep='')
    print()

    # Getting Homography matrices
    h1, w1, _ = data_img1.shape
    h2, w2, _ = data_img2.shape
    H1, H2 = get_homos(data_img1, data_img2)

    # Displaying Epipolar lines
    display_epipolar(data_img1, data_img2)

    img1_rectified = cv.warpPerspective(data_img1, H1, (w1, h1))
    img2_rectified = cv.warpPerspective(data_img2, H2, (w2, h2))
    cv.imshow("Rectified Image 1", img1_rectified)
    cv.imshow("Rectified Image 2", img2_rectified)

    cv.waitKey(0)
    cv.destroyAllWindows()

    # Estimating disparity and depth using SSD
    gray_rect_img1 = cv.cvtColor(img1_rectified, cv.COLOR_BGR2GRAY)
    gray_rect_img2 = cv.cvtColor(img2_rectified, cv.COLOR_BGR2GRAY)
    estimate_disparity_depth()
