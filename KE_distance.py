import numpy as np

def KE_distance(real, fake):
    """
    real: np.array(6, n_test)
    fake: np.array(6, n_test)
    """
    
    s_max = real.shape[0]
    n_test = real.shape[1]
    real = np.sort(real, axis = 1)
    fake = np.sort(fake, axis = 1)
    
    R = np.zeros((s_max, n_test))
    
    for s in range(s_max):
        for i in range(n_test):
            R_i = 0.0
            for j in range(n_test):
                if real[s, j] <= real[s, i]:
                    R_i = R_i + 1
                else:
                    R_i = R_i + 0
            R[s, i] = (1.0 / (n_test - 1.0)) * R_i

    R_tilde = np.zeros((s_max, n_test))
    
    for s in range(s_max):
        for i in range(n_test):
            R_tilde_i = 0.0
            for j in range(n_test):
                if fake[s, j] <= fake[s, i]:
                    R_tilde_i = R_tilde_i + 1
                else:
                    R_tilde_i = R_tilde_i + 0
            R_tilde[s, i] = (1.0 / (n_test - 1.0)) * R_tilde_i
    
    L_D = np.zeros(s_max)
    for s in range(s_max):
        L_D[s] = -n_test
        for i in range(1, n_test+1):
            L_D[s] += (1.0/n_test)* abs(R_i[s,i] - R_tilde_i[s,i])
        
    return L_D