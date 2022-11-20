import numpy as np

def AD_distance(real, fake):
    """
    real: np.array(6, n_test)
    fake: np.array(6, n_test)
    """
    
    s_max = real.shape[0]
    n_test = real.shape[1]
    real = np.sort(real, axis = 1)
    fake = np.sort(fake, axis = 1)
    
    u = np.zeros((s_max, n_test))
    
    for s in range(s_max):
        for i in range(n_test):
            u_i = 0.0
            for j in range(n_test):
                if real[s, j] <= fake[s, i]:
                    u_i = u_i + 1
                else:
                    u_i = u_i + 0
            u[s, i] = (1.0 / (n_test + 2.0)) * (u_i + 1)
    
    W = np.zeros(s_max)
    for s in range(s_max):
        W[s] = -n_test
        for i in range(1, n_test+1):
            w = (2 * i - 1) * (np.log(u[s, i-1]) + np.log(1 - u[s, n_test-i]))
    
            W[s] += -(1.0/n_test)* w
        
    return W