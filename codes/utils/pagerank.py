from numpy import *

def powerMethodBase(A,x0,iter):
 """ basic power method """
 for i in range(iter):
  x0 = dot(A,x0)
  x0 = x0/linalg.norm(x0,1)
 return x0

def powerMethod(A,x0,m,iter):
 """ power method modified to compute
     the maximal real eigenvector 
     of the matrix M built on top of the input matrix A """
 n = A.shape[1]
 delta = m*(array([1]*n,dtype='float64')/n) # array([1]*n is [1 1 ... 1] n times
 for i in range(iter):
  x0 = dot((1-m),dot(A,x0)) + delta
 return x0

def maximalEigenvector(A):
 """ using the eig function to compute eigenvectors """
 n = A.shape[1]
 w,v = linalg.eig(A)
 return abs(real(v[:n,0])/linalg.norm(v[:n,0],1))

def linearEquations(A,m):
 """ solving linear equations 
     of the system (I-(1-m)*A)*x = m*s """
 n = A.shape[1]
 C = eye(n,n)-dot((1-m),A)
 b = m*(array([1]*n,dtype='float64')/n)
 return linalg.solve(C,b)

def getTeleMatrix(A,m):
 """ return the matrix M
     of the web described by A """
 n = A.shape[1]
 S = ones((n,n))/n
 return (1-m)*A+m*S


def centrality_scores(X, alpha=0.85, max_iter=100, tol=1e-10):
    """Power iteration computation of the principal eigenvector

    This method is also known as Google PageRank and the implementation
    is based on the one from the NetworkX project (BSD licensed too)
    with copyrights by:

      Aric Hagberg <hagberg@lanl.gov>
      Dan Schult <dschult@colgate.edu>
      Pieter Swart <swart@lanl.gov>
    """
    n = X.shape[0]
    X = X.copy()
    incoming_counts = np.asarray(X.sum(axis=1)).ravel()

    print("Normalizing the graph")
    for i in incoming_counts.nonzero()[0]:
        X.data[X.indptr[i]:X.indptr[i + 1]] *= 1.0 / incoming_counts[i]
    dangle = np.asarray(np.where(X.sum(axis=1) == 0, 1.0 / n, 0)).ravel()

    scores = np.ones(n, dtype=np.float32) / n  # initial guess
    for i in range(max_iter):
        print("power iteration #%d" % i)
        prev_scores = scores
        scores = (alpha * (scores * X + np.dot(dangle, prev_scores))
                  + (1 - alpha) * prev_scores.sum() / n)
        # check convergence: normalized l_inf norm
        scores_max = np.abs(scores).max()
        if scores_max == 0.0:
            scores_max = 1.0
        err = np.abs(scores - prev_scores).max() / scores_max
        print("error: %0.6f" % err)
        if err < n * tol:
            return scores

    return scores