import numpy as np
from numpy.linalg import inv as inv
import numpy.linalg as ng
import time

def HaLRTC(dense_tensor, sparse_tensor, mask, alpha, rho, maxiter):
    
    dim0 = sparse_tensor.ndim
    dim1, dim2, dim3 = sparse_tensor.shape
    position = np.where(mask == 0.0)
    binary_tensor = mask
    tensor_hat = sparse_tensor.copy()
    
    Z = np.zeros((dim1, dim2, dim3, dim0)) # \boldsymbol{\mathcal{Z}} (n1*n2*3*d)
    T = np.zeros((dim1, dim2, dim3, dim0)) # \boldsymbol{\mathcal{T}} (n1*n2*3*d)
    
    for iters in range(maxiter):
        for k in range(dim0):
            Z[:, :, :, k] = mat2ten(svt(ten2mat(tensor_hat + T[:, :, :, k] / rho, k), 
                                        alpha / rho), np.array([dim1, dim2, dim3]), k)
        tensor_hat = np.mean(Z - T / rho, axis = 3)
        tensor_hat[position] = sparse_tensor[position]
        for k in range(dim0):
            T[:, :, :, k] = T[:, :, :, k] + rho * (tensor_hat - Z[:, :, :, k])
        print ("Iteration Count = " + str(iters))

    return tensor_hat

def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')

def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(tensor_size[index]), order = 'F'), 0, mode)

def svt(mat, lambda0): ## Singular value thresholding (SVT)
    u, s, v = np.linalg.svd(mat, full_matrices = 0)
    vec = s - lambda0
    vec[np.where(vec < 0)] = 0
    
    return np.matmul(np.matmul(u, np.diag(vec)), v)