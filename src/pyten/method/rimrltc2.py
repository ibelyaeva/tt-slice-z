import numpy as np
import math
import pyten.tenclass
from pyten.tools.tenerror import  iteration_cost
import pyten
from pyten.tenclass import Tensor  # Use it to construct Tensor object

def project_and_retract(z, tau):
    m = z.shape[0]
    n = z.shape[1]
    if 2 * m < n:
        [U, Sigma2, V] = np.linalg.svd(np.dot(z, z.T))
        S = np.sqrt(Sigma2)
        tol = np.max(z.shape) * (2 ** int(math.log(max(S), 2))) * 2.2204 * 1E-16
        k = np.sum(S > max(tol, tau))
        mid = [max(S[i] - tau, 0) * 1.0 / S[i] for i in range(k)]
        X = np.dot(np.dot(U[:, 0:k], np.dot(np.diag(mid), U[:, 0:k].T)), z)
        return X, k, Sigma2
    if m > 2 * n:
        z = z.T
        [U, Sigma2, V] = np.linalg.svd(np.dot(z, z.T))
        S = np.sqrt(Sigma2)
        tol = np.max(z.shape) * (2 ** int(math.log(max(S), 2))) * 2.2204 * 1E-16
        k = np.sum(S > max(tol, tau))
        mid = [max(S[i] - tau, 0) * 1.0 / S[i] for i in range(k)]
        X = np.dot(np.dot(U[:, 0:k], np.dot(np.diag(mid), U[:, 0:k].T)), z)
        return X.T, k, Sigma2

    [U, S, V] = np.linalg.svd(z)
    Sigma2 = S ** 2
    k = sum(S > tau)
    X = np.dot(U[:, 0:k], np.dot(np.diag(S[0:k] - tau), V[0:k, :]))
    return X, n, Sigma2


def rimrltc(x, x_true, x_init, omega=None, alpha=None, beta=None, max_iter=100, epsilon=1e-5, printitn=100):

    print("Riemannian Tensor Completion has started")
    T = x.data.copy()
    N = x.ndims
    dim = x.shape
    normX = x.norm()

    if printitn == 0:
        printitn = max_iter
    if omega is None:
        omega = x.data * 0 + 1

    if alpha is None:
        alpha = np.ones([N])
        alpha = alpha / sum(alpha)
    
    print ("Alpha: " +str(alpha))
    if beta is None:
        beta = 1e-6

    # Initialization
    #x = Tensor(x_init)
    x.data[omega == 0] = np.mean(x.data[omega == 1])
    errList = np.zeros([max_iter, 1])
    errTest = np.zeros([max_iter, 1])

    Y = range(N)
    M = range(N)

    for i in range(N):
        Y[i] = x.data
        M[i] = np.zeros(dim)

    Msum = np.zeros(dim)
    Ysum = np.zeros(dim)

    for k in range(max_iter):

        if (k + 1) % printitn == 0 and k != 0 and printitn != max_iter:
            print 'RimLRTC: iterations = {0}  difference = {1}\n'.format(k, errList[k - 1])

        beta = beta * 1.05

        # Update Y
        Msum = 0 * Msum
        Ysum = 0 * Ysum
        for i in range(N):
            A = pyten.tenclass.Tensor(x.data - M[i] / beta)
            temp = pyten.tenclass.Tenmat(A, i + 1)
            [temp1, tempn, tempSigma2] = project_and_retract(temp.data, alpha[i] / beta)
            temp.data = temp1
            Y[i] = temp.totensor().data
            Msum = Msum + M[i]
            Ysum = Ysum + Y[i]
            
            print 'RimLRTC: iterations = {0}  difference = {1}\n'.format(k, errList[k - 1])
            

        # update x
        Xlast = x.data.copy()
        Xlast = pyten.tenclass.Tensor(Xlast)
        x.data = (Msum + beta * Ysum) / (N * beta)
        x.data = T * omega + x.data * (1 - omega)

        # Update unfolding matrices
        for i in range(N):
            M[i] = M[i] + beta * (Y[i] - x.data)

        # Compute the error
        diff = x.data - Xlast.data
        errList[k] = np.linalg.norm(diff) / normX

        
        test_error = iteration_cost(x, x_true, omega)
        errTest[k] = test_error
        
        if k>1:
            diff_train_cost = errList[k] - errList[k-1]
        else:
            diff_train_cost = 0
            
            
        x_true_omega = np.linalg.norm(x_true.data * omega)
        diff_on_omega = np.linalg.norm((x.data - x_true.data) * omega)
        rse_train = diff_on_omega/x_true_omega
               
        print ("Iteration #: " +str(k) + ";Train Error: " + str(errList[k]) + "; Test Error: " + str(test_error) + "; Relative Error Per Training Iteration: " + str(diff_train_cost) + "; RSE on Omega: " + str(rse_train))
        
        if errList[k] < epsilon:
            errList = errList[0:(k + 1)]
            errTest = errTest[0:(k + 1)]
            break


    print 'RimRTC completed: total iterations = {0}   difference = {1}\n\n'.format(k + 1, errList[k])
    return x, errList, errTest

def get_alpha(X):
    n = X.n
    d = len(n)
    alpha = np.zeros(d-1)
    beta = np.zeros(d-1)
    delta = np.zeros(d-1)
    for k in xrange(d-1):
        first = 1.
        second = 1.
        for l in xrange(k+1):
            first *= n[l]
        for l in xrange(k+1, d):
            second *= n[l]
        #print first, second
        #print np.minimum(first, second)
        delta[k] = np.minimum(first, second)
    alphas = delta/np.sum(delta)
    return alphas
