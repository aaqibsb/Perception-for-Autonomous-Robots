from pandas import *
import numpy as np
import matplotlib.pyplot as plt

# RANSAC function - k=iterations; t=threshold value; d=close data points to assert that model fits well to data
def ransac(data, k, t, d):
    best_model = None
    best_error = float('inf')

    x = data[data.columns[0]].to_numpy()
    y = data[data.columns[1]].to_numpy()
    z = data[data.columns[2]].to_numpy()

    data_arr = np.vstack((x, y, z)).T

    for i in range(k):
        sample = data.sample(n=3)

        x1 = sample[sample.columns[0]].to_numpy()
        y1 = sample[sample.columns[1]].to_numpy()
        z1 = sample[sample.columns[2]].to_numpy()

        sample_arr = np.vstack((x1, y1, z1)).T

        hyp_model = least_square(sample)
        inliers = []

        for point in data_arr:
            if point not in sample_arr:
                if distance(point, hyp_model) < t:
                    inliers.append(point)

        inliers_df = DataFrame(inliers)
        better_model_df = concat([sample,inliers_df])

        if len(inliers) > d:
            better_model = least_square(better_model_df)
            curr_error = error(data_arr, better_model)

            if curr_error < best_error:
                best_model = better_model
                best_error = curr_error
    return best_model

# Error calculation for each data point
def distance(point, hyp_model):
    A = hyp_model[0,0]
    B = hyp_model[1,0]
    C = hyp_model[2,0]
    z = point[2]

    Z = A*point[0] + B*point[1] + C
    dist = (Z-z)**2

    return dist

# Mean error calculation to compare models
def error(data, hyp_model):
    errors = [distance(point, hyp_model) for point in data]
    return np.mean(errors)

# Standard Least Square function
def least_square(data):
    x = data[data.columns[0]].to_numpy()
    y = data[data.columns[1]].to_numpy()
    z = data[data.columns[2]].to_numpy()

    z.shape = len(z),1
    X = np.vstack((x, y, [1]*len(x))).T
    Y = z

    B = np.linalg.inv(X.T @ X) @ (X.T @ Y)
    return B

# Total Least Square function
def total_least_square(data):
    x = data[data.columns[0]].to_numpy()
    y = data[data.columns[1]].to_numpy()
    z = data[data.columns[2]].to_numpy()

    X = np.vstack((x, y, z)).T
    u = X.mean(axis=0)
    U = (X - u).T.dot(X - u)

    eigenvalues, eigenvectors = np.linalg.eig(U)

    smallest_eigenvalue_index = np.argmin(eigenvalues)

    A = eigenvectors[:, smallest_eigenvalue_index]
    return A


if __name__ == "__main__":
    see = read_csv(r"CSV/pc1.csv", header=None)

    x = see[see.columns[0]].to_numpy()
    y = see[see.columns[1]].to_numpy()
    z = see[see.columns[2]].to_numpy()

    """ Plotting for data points """
    ax = plt.axes(projection="3d")
    ax.scatter3D(x, y, z, color="green")

    pc1_ransac = ransac(see, 100, 1, d=100)

    pc1_TLS = total_least_square(see)

    pc1_LS = least_square(see)

    x_1 = np.linspace(-10, 10, 10)
    y_1 = np.linspace(-10, 10, 10)

    X, Y = np.meshgrid(x_1, y_1)

    """ Total Least Squares """
    # d = pc1_TLS[0]*see[0].mean() + pc1_TLS[1]*see[1].mean() + pc1_TLS[2]*see[2].mean()
    # Z = (-1/pc1_TLS[2])*(pc1_TLS[0]*X + pc1_TLS[1]*Y - d)
    #
    # ax.plot_surface(X, Y, Z, alpha=0.5)
    # plt.title("Total Least Squares")
    # plt.show()

    """ Least Squares """
    # Z = pc1_LS[0,0]*X + pc1_LS[1,0]*Y + pc1_LS[2,0]
    #
    # surf = ax.plot_surface(X, Y, Z, alpha=0.5)
    # plt.title("Standard Least Squares")
    # plt.show()

    """ RANSAC """
    Z = pc1_ransac[0,0]*X + pc1_ransac[1,0]*Y + pc1_ransac[2,0]

    ax.plot_surface(X, Y, Z, alpha=0.5)
    plt.title("RANSAC")
    plt.show()


