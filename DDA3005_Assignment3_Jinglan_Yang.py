import numpy as np
import scipy.linalg

def wilkinson_matrix(n):
    """Generate the Wilkinson matrix of size n."""
    W = np.zeros((n, n))
    for i in range(n):
        W[i, i] = 1  # Diagonal elements
        W[i,n-1] = 1
        for j in range(i):
            W[i, j] = -1  # Sub-diagonal
    return W

def inverse_wilkinson_matrix(n):
    W_inverse = np.zeros((n, n))
    for i in range(n):
        if i==n-1:
            W_inverse[i, i] = (0.5)**(i)
            for j in range(i):
                W_inverse[i,j]=(0.5)**(j+1)
        else:
            W_inverse[i, i] = 0.5  # Diagonal elements
        for j in range(i+1,n):
            if j!=n-1:
                W_inverse[i, j] = -(0.5)**(j-i+1) # Super-diagonal
            else:
                W_inverse[i, j] = -(0.5)**(j-i)

    return W_inverse

def compute_errors(n, num_trials):
    lu_errors = []
    qr_errors = []

    for _ in range(num_trials):
        # Generate the Wilkinson matrix W and random vector b
        W = wilkinson_matrix(n)
        W_inverse=inverse_wilkinson_matrix(n)
        b = np.random.normal(0, 1, n)

        # Compute the true solution using the inverse
        #x_star = np.linalg.solve(W, b)
        x_star=np.dot(W_inverse,b)

        # LU Decomposition
        P, L, U = scipy.linalg.lu(W)
        y = scipy.linalg.solve(L, b)
        x_lu = scipy.linalg.solve(U, y)

        # QR Decomposition
        Q, R = np.linalg.qr(W)
        y_qr = Q.T @ b
        x_qr = scipy.linalg.solve(R, y_qr)

        # Compute forward errors
        lu_error = np.linalg.norm(x_star - x_lu) / np.linalg.norm(x_star)
        qr_error = np.linalg.norm(x_star - x_qr) / np.linalg.norm(x_star)

        lu_errors.append(lu_error)
        qr_errors.append(qr_error)

    return lu_errors, qr_errors

# Experiment parameters
n_values = [10,50,75,100,250,500,1000]  # Different sizes of W
num_trials = 100  # Number of trials for each size

LU_avg_error=[]
QR_avg_error=[]

# Run experiments
for n in n_values:
    lu_errors, qr_errors = compute_errors(n, num_trials)
    print(f"n = {n}:")
    print(f"  LU average error: {np.mean(lu_errors)}")
    LU_avg_error.append(np.mean(lu_errors))
    print(f"  QR average error: {np.mean(qr_errors)}")
    QR_avg_error.append(np.mean(qr_errors))

for i in range(len(n_values)):
    print("n= ",n_values[i])
    print("LU forward error: ",LU_avg_error[i])
    print("QR forward error: ",QR_avg_error[i])

