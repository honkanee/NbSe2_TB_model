import numpy as np
import matplotlib.pyplot as plt
import Hamilton_NbSe2

SC_Delta1 = 0.01
SC_Delta2 = 0.01
SC_Delta3 = 0.01
SC_Delta4 = 0.01

factor = 1

def Nambu_Gorgov_matrix(k):

    H_NbSe2 = Hamilton_NbSe2.Hamilton_MoS2(k, SC_Delta1, SC_Delta2, SC_Delta3, SC_Delta4, E0=1.3)
    H_MoS2 = Hamilton_NbSe2.Hamilton_MoS2(k, E0=0)

    Z = np.zeros(np.shape(H_NbSe2))

    A = np.zeros((11,11))
    A[10,2] = factor*Hamilton_NbSe2.V_pps

    A = Hamilton_NbSe2.base_switch(A)
    A = np.block([
        [A, np.zeros((11,11))],
        [np.zeros((11,11)), A]
    ])
    A = Hamilton_NbSe2.apply_SC(A, 0, 0, 0, 0)

    H = np.block([
        [H_NbSe2, A, Z],
        [A.T, H_MoS2, A],
        [Z, A.T, H_NbSe2]
        ])

    return H

def main():

    H = Nambu_Gorgov_matrix([0,0])

    print(np.shape(H))

        # Ridolfi
    K = np.array([2*np.pi/3,-2*np.pi/np.sqrt(3)])
    M = np.array([np.pi,-np.pi/np.sqrt(3)])

    Gamma = np.zeros(2)

    K_dist = np.linalg.norm(K)
    M_dist = np.linalg.norm(M)
    K_M_dist = np.linalg.norm(K-M)
    dist_sum = K_dist+M_dist+K_M_dist

    N = 1000
    Gamma_M = np.linspace(Gamma, M, round(N*M_dist/dist_sum))
    M_K = np.linspace(M,K,round(N*K_M_dist/dist_sum))
    K_Gamma = np.linspace(K,Gamma,round(N*K_dist/dist_sum))
    k_values = np.concatenate((Gamma_M, M_K, K_Gamma))

    n_base = 132
    E_values = np.zeros((np.shape(k_values)[0], n_base))
    cs = np.zeros((np.shape(k_values)[0], n_base, n_base), dtype="complex")
    cumulative_H = np.zeros((n_base,n_base))
    for i in range(np.shape(k_values)[0]):
        k = k_values[i]
        H = Nambu_Gorgov_matrix(k)
        E, c = np.linalg.eigh(H)

        E_values[i] = E
        cs[i] = c

        cumulative_H += np.abs(H)


    # Bands
    plt.figure()
    for i in range(n_base):
        plt.plot(E_values[:,i], color="black")

    plt.xticks([0, round(N*M_dist/dist_sum), round(N*(M_dist+K_M_dist)/dist_sum),N], [r"$\Gamma$","M","K",r"$\Gamma$"])
    plt.xlabel("k")
    plt.ylabel("E [eV]")
    plt.axvline(x=round(N*M_dist/dist_sum), color='black', linestyle='--')
    plt.axvline(x=round(N*(K_M_dist+M_dist)/dist_sum), color='black', linestyle='--')
    #plt.ylim(-2,2.5)
    plt.axhline()

    plt.show()


    return

if __name__=="__main__":
    main()