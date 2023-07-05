import numpy as np
import matplotlib.pyplot as plt
from Hamilton_NbSe2 import Hamilton_MoS2

INCLUDE_SOC = True
INCLUDE_SC = True

def main():
    
    # Ridolfi
    K = np.array([2*np.pi/3,-2*np.pi/np.sqrt(3)])
    M = np.array([np.pi,-np.pi/np.sqrt(3)])

    Gamma = np.zeros(2)

    K_dist = np.linalg.norm(K)
    M_dist = np.linalg.norm(M)
    K_M_dist = np.linalg.norm(K-M)
    dist_sum = K_dist+M_dist+K_M_dist

    N = 2000
    Gamma_M = np.linspace(Gamma, M, round(N*M_dist/dist_sum))
    M_K = np.linspace(M,K,round(N*K_M_dist/dist_sum))
    K_Gamma = np.linspace(K,Gamma,round(N*K_dist/dist_sum))
    k_values = np.concatenate((Gamma_M, M_K, K_Gamma))

    n_base = 11
    if INCLUDE_SOC:
        n_base *= 2
    if INCLUDE_SC:
        n_base *= 2
    E_values = np.zeros((np.shape(k_values)[0], n_base))
    cs = np.zeros((np.shape(k_values)[0], n_base, n_base), dtype="complex")
    cumulative_H = np.zeros((n_base,n_base))
    eta = 0.01
    for i in range(np.shape(k_values)[0]):
        k = k_values[i]
        H = Hamilton_MoS2(k)
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
    plt.ylim(-2,2.5)
    plt.axhline()

    # Spectrums
    if True:
        Es = np.linspace(-1,1,500)
        orbitals = np.array([[2, 35], [1, 34], [0, 33]])
        txt = [r"e $d_{z^2 \uparrow}$", r"e $d_{x^2 \uparrow}$", r"e $d_{xy \uparrow}$", r"e $p_{x,S \uparrow}$", r"e $p_{y,S \uparrow}$", r"e $p_{z,A \uparrow}$", r"e $p_{d_{xz} \uparrow}$", r"e $d_{yz \uparrow}$", r"e $p{x,A \uparrow}$", r"e $p_{y,A \uparrow}$", r"e $p_{z,S \uparrow}$",
            r"e $d_{z^2 \downarrow}$", r"e $d_{x^2 \downarrow}$", r"e $d_{xy \downarrow}$", r"e $p_{x,S \downarrow}$", r"e $p_{y,S \downarrow}$", r"e $p_{z,A \downarrow}$", r"e $p_{d_{xz} \downarrow}$", r"e $d_{yz \downarrow}$", r"e $p{x,A \downarrow}$", r"e $p_{y,A \downarrow}$", r"e $p_{z,S \downarrow}$",
            r"h $d_{z^2 \uparrow}$", r"h $d_{x^2 \uparrow}$", r"h $d_{xy \uparrow}$", r"h $p_{x,S \uparrow}$", r"h $p_{y,S \uparrow}$", r"h $p_{z,A \uparrow}$", r"h $p_{d_{xz} \uparrow}$", r"h $d_{yz \uparrow}$", r"h $p{x,A \uparrow}$", r"h $p_{y,A \uparrow}$", r"h $p_{z,S \uparrow}$",
            r"h $d_{z^2 \downarrow}$", r"h $d_{x^2 \downarrow}$", r"h $d_{xy \downarrow}$", r"h $p_{x,S \downarrow}$", r"h $p_{y,S \downarrow}$", r"h $p_{z,A \downarrow}$", r"h $p_{d_{xz} \downarrow}$", r"h $d_{yz \downarrow}$", r"h $p{x,A \downarrow}$", r"h $p_{y,A \downarrow}$", r"h $p_{z,S \downarrow}$",
            ]
        n_orbitals = np.shape(orbitals)[0]
        DOS = np.zeros((n_orbitals, np.shape(Es)[0], np.shape(k_values)[0]),dtype='complex')
        eta = 0.1
        for ii in range(n_orbitals):
            for i in range(np.shape(k_values)[0]):
                for j in range(np.shape(Es)[0]):
                    E = Es[j]
                    DOS[ii,j,i] = -1/np.pi*np.sum(cs[i,orbitals[ii,0],:]*np.conj(cs[i,orbitals[ii,1],:])/(E-E_values[i,:]+1j*eta))
            print(ii+1)

        plt.figure()
        X, Y = np.meshgrid(range(np.shape(k_values)[0]),Es)
        for i in range(n_orbitals):
            plt.subplot(2,3,i+1)
            plt.pcolormesh(X,Y, np.abs(np.imag(DOS[i])))
            plt.ylabel("E")
            plt.xlabel("k")
            plt.xticks([0, round(N*K_dist/dist_sum), round(N*(M_dist+K_M_dist)/dist_sum),N], [r"$\Gamma$","M","K",r"$\Gamma$"])
            plt.title("Im " + txt[orbitals[i,0]] + " " + txt[orbitals[i,1]])
            plt.subplot(2,3,i+4)
            plt.pcolormesh(X,Y, np.abs(np.real(DOS[i])))
            """
            plt.plot(E_values[:,12], "r--")
            plt.plot(E_values[:,13], "r--")
            plt.plot(E_values[:,14], "r--")
            plt.plot(E_values[:,15], "r--")
            """
            plt.title("Re " + txt[orbitals[i,0]] + " " + txt[orbitals[i,1]])
            plt.ylabel("E")
            plt.xlabel("k")
            plt.xticks([0, round(N*K_dist/dist_sum), round(N*(M_dist+K_M_dist)/dist_sum),N], [r"$\Gamma$","M","K",r"$\Gamma$"])
        plt.tight_layout()

    # Green's function matrix elements
    if False:
        Es = np.linspace(-1,1,250)
        eta = 0.1
        G_2_35 = np.zeros((np.shape(k_values)[0], np.shape(Es)[0]), dtype='complex')
        G_1_34 = np.zeros((np.shape(k_values)[0], np.shape(Es)[0]), dtype='complex')
        G_0_33 = np.zeros((np.shape(k_values)[0], np.shape(Es)[0]), dtype='complex')
        G = np.zeros((n_base,n_base), dtype='complex')
        cumulative_G = np.zeros((n_base,n_base))
        for i in range(np.shape(k_values)[0]):
            for j in range(np.shape(Es)[0]):
                k = k_values[i]
                H = Hamilton_MoS2(k)
                E, c = np.linalg.eigh(H)
                G = np.linalg.inv((Es[j]+1j*eta)*np.identity(n_base)-H)
                G_2_35[i,j] = G[2,35]
                G_1_34[i,j] = G[1,34]
                G_0_33[i,j] = G[0,33]
                cumulative_G += np.abs(G)
            print(i)


        plt.figure()
        X, Y = np.meshgrid(range(np.shape(k_values)[0]),Es)
        #elements = np.array([[2,35], [3,36], [0,34]])
        elements = np.array([G_2_35, G_3_36, G_0_34])
        for i in range(len(elements)):
            plt.subplot(2,3,i+1)
            plt.pcolormesh(X,Y, np.abs(np.imag(elements[i]).T))
            plt.subplot(2,3,i+4)
            plt.pcolormesh(X,Y, np.abs(np.real(elements[i]).T))
            #plt.pcolormesh(X,Y, np.imag(Gs[:,:,elements[i,0],elements[i,1]].T))
            #plt.title(txt[orbitals[i,0]] + " " + txt[orbitals[i,1]])
            plt.ylabel("E")
            plt.xlabel("k")
            plt.xticks([0, round(N*K_dist/dist_sum), round(N*(M_dist+K_M_dist)/dist_sum),N], [r"$\Gamma$","M","K",r"$\Gamma$"])
        plt.tight_layout()

    plt.figure()
    plt.plot(k_values[:,0], k_values[:,1])

    plt.show()
    return


if __name__=="__main__":
    main()