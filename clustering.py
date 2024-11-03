import numpy as np
import time
import matplotlib.pyplot as plt

def clustering(X, Z0 = None, opts = {'iter': 100, 'display': True}):
    """
    ----------------------------------------------------------------------------
    INPUT
    ----------------------------------------------------------------------------
      - X     (n x m) matrix containing the m different data points.
      - Z0    (n x k) initial points for the clusters (will be estimated by
                      Z0 = rand(n,3) if left empty).
      - opts  dict of options
                  iter       number of maximum iterations (default = 100).
                  display    print total distances (default = true).
    """
    n, m = X.shape
    print('n=',n,'m=',m)
    # --------------------------------------------------------------------------
    # PROCESS INPUT AND OPTIONS
    # --------------------------------------------------------------------------

    if Z0 is None:
        k = 3
        Z0 = np.random.randn(n, k)
        #print("Z0:",Z0)
    else:
        k = Z0.shape[1]
    Z = Z0

    opts_alg = {}
    opts_alg['iter'] = opts.get('iter', 100)
    opts_alg['display'] = opts.get('display', True)

    sX = np.tile(X, (k, 1))

    # --------------------------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------------------------
    #cpu_times = []
    for i in range(opts_alg['iter']):
        #start_time = time.time()
        # calculate distances
        sZ = np.abs(sX - np.tile(Z.flatten('F').reshape(n*k, 1, order='F'), (1, m))) ** 2
        sZ = np.sqrt(sum(sZ.reshape(n, m*k, order='F')))
        # select minimum distances
        sZ = sZ.reshape(k, m, order='F')
        mZ = np.min(sZ, axis=0)
        c = np.argmin(sZ, axis=0)

        # calculate new means
        for j in range(k):
            ind = (c == j)
            #Z[:, j] = np.mean(X[:, ind], axis=1)
            if np.any(ind):  # Only calculate mean if the cluster has points
                Z[:, j] = np.mean(X[:, ind], axis=1)
            else:
                Z[:, j] = np.random.randn(n)  # Handle empty clusters
        #elapsed_time = time.time() - start_time
        #cpu_times.append(elapsed_time)

    # output
    if opts_alg['display']:
        s1 = np.sum(np.sum(mZ ** 2))
        s2 = s1 / (m*k)
        #print(s1,s2)
        print('Total sum of distances = {:5.2f}; Weighted sum of distances = {:1.6e}'.format(s1, s2))
    #print("Z:", Z)
    return c, Z, cpu_times

# Function to generate data clouds
def generate_data_clouds(p):
    r = np.random.normal(0, 1, (p, 6))  # Generate random numbers

    A = r[:, :2] 
    B = 4 * r[:, 2:4] - np.array([[4, 6]]) 
    C = 2 * r[:, 4:6] - np.array([[5, 2]]) 

    return A, B, C

# Measure CPU time for different p values
p_values = [100 * i for i in range(1, 6)]  # or use range(1, 11) for larger sizes
cpu_times = []

for p in p_values:    
    A, B, C = generate_data_clouds(p)
    X = np.concatenate((A, B, C), axis=0) 
    #Z0=np.array([[0, 0],[-4,-6],[-5,-2]]) 
    start_time = time.time()
    c,Z,cpu_times=clustering(X, Z0=None, opts = {'iter': 1000, 'display': True})
    elapsed_time = time.time() - start_time
    cpu_times.append(elapsed_time/1000)


# Plotting the results
plt.plot(p_values, cpu_times, marker='o')
plt.xlabel('Data Sizes')
plt.ylabel('Mean CPU Time (seconds)')
plt.title('Mean CPU Time per Iteration for Different Data Sizes')
#plt.legend()
plt.grid()
plt.show()

# Visualizing the clustering for the largest p
#print(c)
#print(Z)

A_cluster=[]
B_cluster=[]
C_cluster=[]
for i in range(Z.shape[0]):
    if np.argmin(Z[i])==0:
        A_cluster.append(X[i])
    elif np.argmin(Z[i])==1:
        B_cluster.append(X[i])
    elif np.argmin(Z[i])==2:
        C_cluster.append(X[i])
A_cluster=np.array(A_cluster)
B_cluster=np.array(B_cluster)
C_cluster=np.array(C_cluster)
#print(A_cluster)
plt.scatter(A_cluster[:, 0],A_cluster[:, 1], c='red', label='Cluster A')
plt.scatter(B_cluster[:, 0], B_cluster[:, 1], c='blue', label='Cluster B')
plt.scatter(C_cluster[:, 0], C_cluster[:, 1], c='green', label='Cluster C')
plt.scatter(np.mean(A_cluster[:, 0]), np.mean(A_cluster[:, 1]), c='black', marker='x', s=100, label='Centroid A')
plt.scatter(np.mean(B_cluster[:, 0]), np.mean(B_cluster[:, 1]), c='black', marker='x', s=100, label='Centroid B')
plt.scatter(np.mean(C_cluster[:, 0]), np.mean(C_cluster[:, 1]), c='black', marker='x', s=100, label='Centroid C')
plt.legend()
plt.title('Clustering Result for Largest p')
plt.show()


