from pandas import *
import numpy as np

# Covariance function
def covariance(file):
    x1 = file[file.columns[0]].to_numpy()
    x2 = file[file.columns[1]].to_numpy()
    x3 = file[file.columns[2]].to_numpy()

    X = np.vstack((x1, x2, x3)).T
    u = X.mean(axis=0)
    n = X.shape[0]

    Cov = (1 / n) * (X - u).T.dot(X - u)

    return Cov


if __name__ == "__main__":
    see = read_csv(r"CSV/pc1.csv", header=None)
    cov_matrix = covariance(see)
    print("Covariance matrix: ")
    print(cov_matrix)

    print()

    # Computing the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    print("Eigen values:")
    print(eigenvalues)

    print()
    print("Eigen vectors:")
    print(eigenvectors)

    # Finding index of the eigenvector with the smallest eigenvalue
    smallest_eigenvalue_index = np.argmin(eigenvalues)

    # Extracting the surface normal (eigenvector with smallest eigenvalue)
    surface_normal = eigenvectors[:, smallest_eigenvalue_index]

    # Computing the magnitude of the surface normal using norm
    surface_normal_magnitude = np.linalg.norm(surface_normal)

    print()
    print("Surface normal:", surface_normal)
    print("Magnitude:", surface_normal_magnitude)
