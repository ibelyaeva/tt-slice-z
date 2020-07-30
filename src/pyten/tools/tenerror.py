import numpy as np
import pyten.tenclass
import copy


def tenerror(x_hat, x_true, omega):
    
    """
    Calculate Two Type Kinds of Error
    :param fitx: fitted tensor
    :param realx: ground-truth tensor
    :param omega: index tensor of observed entries
    """

    if type(omega) != np.ndarray and type(omega) != pyten.tenclass.Tensor:
        raise ValueError("Cannot recognize the format of observed Tensor!")
    elif type(omega) == pyten.tenclass.Tensor:
        omega = omega.tondarray
    if type(x_true) == np.ndarray:
        x_true = pyten.tenclass.Tensor(x_true)
    if type(x_hat) == np.ndarray:
        x_true = pyten.tenclass.Tensor(x_true)
        
    norm1 = np.linalg.norm(x_true.data)
    
    abs_err = np.linalg.norm(x_hat.data - x_true.data)  # Absolute Error
    rel_error = abs_err / norm1  # Relative Error

    return abs_err, rel_error

def tenerror_omega(x_hat, x_true, omega):
    
    """
    Calculate Two Type Kinds of Error
    :param fitx: fitted tensor
    :param realx: ground-truth tensor
    :param omega: index tensor of observed entries
    """

    if type(omega) != np.ndarray and type(omega) != pyten.tenclass.Tensor:
        raise ValueError("Cannot recognize the format of observed Tensor!")
    elif type(omega) == pyten.tenclass.Tensor:
        omega = omega.tondarray
    if type(x_true) == np.ndarray:
        x_true = pyten.tenclass.Tensor(x_true)
    if type(x_hat) == np.ndarray:
        x_true = pyten.tenclass.Tensor(x_true)
        
    norm1 = np.linalg.norm(x_true.data)
    
    abs_err = np.linalg.norm(x_hat.data - x_true.data)  # Absolute Error
    rel_error = abs_err / norm1  # Relative Error
    
    norm2 = np.linalg.norm(x_true.data * (1 - omega))
    err2 = np.linalg.norm((x_hat.data - x_true.data) * (1 - omega))

    completion_score  = err2 / norm2
    omega_complement_size = np.count_nonzero(omega)
    omega_complement_size_sqrt = np.sqrt(omega_complement_size)
    print ("Non-Zero Count: " + str(omega_complement_size))
    
    max_true_omega = np.max(x_true.data * (1 - omega))
    min_true_omega = np.min(x_true.data * (1 - omega))
    diff_max = max_true_omega - min_true_omega
    
    print ("omega_complement_size_sqrt: " + str(omega_complement_size_sqrt))
    
    print ("max_true_omega: " + str(max_true_omega))
    print ("min_true_omega: " + str(min_true_omega))
    print ("diff_max: " + str(diff_max))
    
    nrmse = err2/((omega_complement_size_sqrt)*diff_max)
    print ("nrmse: " + str(nrmse))
    
    return abs_err, rel_error, completion_score, nrmse

def iteration_cost(x_hat, x_true, omega):
    
    if type(omega) != np.ndarray and type(omega) != pyten.tenclass.Tensor:
        raise ValueError("Cannot recognize the format of observed Tensor!")
    elif type(omega) == pyten.tenclass.Tensor:
        omega = omega.tondarray
    if type(x_true) == np.ndarray:
        x_true = pyten.tenclass.Tensor(x_true)
    if type(x_hat) == np.ndarray:
        x_true = pyten.tenclass.Tensor(x_true)
        
    norm1 = np.linalg.norm(x_true.data)
    abs_err = np.linalg.norm(x_hat.data - x_true.data)  # Absolute Error
    test_error = abs_err / norm1  # Relative Error
    
    return test_error
        

def tsc_score(x_hat, x_true, omega):
    ten_omega_complement = np.ones_like(omega)
    ten_omega_complement[omega==1] = 0
    x_true_complement_omega = copy.deepcopy(x_true)
    x_true_complement_omega[ten_omega_complement==0]=0.0
    x_hat_compelement = copy.deepcopy(x_hat)
    x_hat_compelement[ten_omega_complement==0]=0.0
    denom = np.linalg.norm(x_true_complement_omega)
    nomin = np.linalg.norm(x_hat_compelement - x_true_complement_omega)
    tsc = nomin/denom
    return tsc
    
def cost(x_hat, x_true, omega):
    
    if type(omega) != np.ndarray and type(omega) != pyten.tenclass.Tensor:
        raise ValueError("Cannot recognize the format of observed Tensor!")
    elif type(omega) == pyten.tenclass.Tensor:
        omega = omega.tondarray
    if type(x_true) == np.ndarray:
        x_true = pyten.tenclass.Tensor(x_true)
    if type(x_hat) == np.ndarray:
        x_true = pyten.tenclass.Tensor(x_true)
        
    cost = 0.5*np.linalg.norm(x_hat.data - x_true.data)**2
    
    return cost


def tsc(x_hat,x_true, mask):
    ten_ones = np.ones_like(mask)
    
    nomin = np.linalg.norm(np.multiply((ten_ones - mask), (x_true - x_hat)))
    denom = np.linalg.norm(np.multiply((ten_ones - mask), x_true))
    score = nomin/denom
    return score