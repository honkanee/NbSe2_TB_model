"""
Script for plotting of bands and state spectrums along a path in Brillouin zone
"""

import numpy as np
import matplotlib.pyplot as plt
from Hamilton_NbSe2 import Hamilton_MoS2
from NbSe2_Fermi_surface import BZ_corners

ITERATE_DELTA = False

Delta1 = 0.01
Delta2 = 0.01
Delta3 = 0.01
Delta4 = 0.01

dirname="./kuvat/Bands"
textstr = '\n'.join((
r'$\Delta_1=%.0f$ meV' % (Delta1*1000, ),
r'$\Delta_2=%.0f$ meV' % (Delta2*1000, ),
r'$\Delta_3=%.0f$ meV' % (Delta3*1000, ),
r'$\Delta_4=%.0f$ meV' % (Delta4*1000)))

def iterate_Delta_matrix(k, iters):
    """
    A try of solving iteratively k-dependent Delta matrix, didn't work
    """
    limit = 0.0001
    eta = 0.001
    H0 = Hamilton_MoS2(k, Delta1, Delta2, Delta3, Delta4)
    Delta_M = H0[:22,22:]
    G = np.linalg.inv(1j*eta*np.identity(44)-H0)
    F = G[:22,22:]
    err = np.zeros(iters)
    for i in range(iters):
        Delta_M_old = Delta_M
        U = Delta_M**2
        Delta_M = -U*F
        H = Hamilton_MoS2(k,Delta_M=Delta_M)
        G = np.linalg.inv(1j*eta*np.identity(44)-H)
        F = G[:22,22:]
        e = abs(np.sum(Delta_M_old**2-Delta_M**2))
        err[i] = e
        if i > 5 and e < limit:
            print(abs(np.sum(Delta_M_old**2-Delta_M**2)))
            break
        print("iteraatio: " + str(i))
    return Delta_M, err

def main():

    N_points = 2000
    
    K = np.array([2*np.pi/3,-2*np.pi/np.sqrt(3)])
    M = np.array([np.pi,-np.pi/np.sqrt(3)])
    Gamma = np.zeros(2)

    K_dist = np.linalg.norm(K)
    M_dist = np.linalg.norm(M)
    K_M_dist = np.linalg.norm(K-M)
    dist_sum = K_dist+M_dist+K_M_dist

    Gamma_M = np.linspace(Gamma, M, round(N_points*M_dist/dist_sum))
    M_K = np.linspace(M,K,round(N_points*K_M_dist/dist_sum))
    K_Gamma = np.linspace(K,Gamma,round(N_points*K_dist/dist_sum))
    k_values = np.concatenate((Gamma_M, M_K, K_Gamma))

    n_base = 44
    Delta_matrices = np.zeros((np.shape(k_values)[0], 22, 22), dtype="complex")
    iters = 100
    covergence = np.zeros((np.shape(k_values)[0], iters))
    E_values = np.zeros((np.shape(k_values)[0], n_base))
    cs = np.zeros((np.shape(k_values)[0], n_base, n_base), dtype="complex")
    cumulative_H = np.zeros((n_base,n_base))
    eta = 0.01
    for i in range(np.shape(k_values)[0]):
        k = k_values[i]
        H = Hamilton_MoS2(k, Delta1, Delta2, Delta3, Delta4)

        if ITERATE_DELTA:
            Delta_matrices[i], covergence[i] = iterate_Delta_matrix(k, iters)
            H = Hamilton_MoS2(k, Delta_M=Delta_matrices[i])
        
        E, c = np.linalg.eigh(H)
        E_values[i] = E
        cs[i] = c

        cumulative_H += np.abs(H)

    if ITERATE_DELTA:
        plt.figure()
        for i in range(np.shape(covergence)[0]):
            plt.plot(covergence[i])
        plt.xlabel("iteraatio")
        plt.ylabel("konvergenssi")

        plt.figure()
        for i in range(22):
            for j in range(22):
                plt.plot(np.abs(Delta_matrices[:,i,j]))
        plt.xticks([0, round(N_points*K_dist/dist_sum), round(N_points*(M_dist+K_M_dist)/dist_sum),N_points], [r"$\Gamma$","M","K",r"$\Gamma$"])
        plt.xlabel("k")
        plt.ylabel("|$F_{i,j}$|")

    # Bands
    plt.figure()
    for i in range(n_base):
        plt.plot(E_values[:,i], color="black")

    plt.xticks([0, round(N_points*M_dist/dist_sum), round(N_points*(M_dist+K_M_dist)/dist_sum),N_points], [r"$\Gamma$","M","K",r"$\Gamma$"])
    plt.xlabel("k")
    plt.ylabel("E [eV]")
    plt.axvline(x=round(N_points*M_dist/dist_sum), color='black', linestyle='--')
    plt.axvline(x=round(N_points*(K_M_dist+M_dist)/dist_sum), color='black', linestyle='--')
    plt.ylim(-0.5,0.5)
    plt.axhline()

    ax = plt.gca()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    
    fig = plt.gcf()
    fig.set_size_inches(8, 8)

    # Spectrums
    Es = np.linspace(-0.5,0.5,1000)
    orbitals = np.array([[2, 35], [1, 34], [0, 33]])
    txt = [r"e $d_{z^2 \uparrow}$", r"e $d_{x^2 \uparrow}$", r"e $d_{xy \uparrow}$", r"e $p_{x,S \uparrow}$", r"e $p_{y,S \uparrow}$", r"e $p_{z,A \uparrow}$", r"e $p_{d_{xz} \uparrow}$", r"e $d_{yz \uparrow}$", r"e $p{x,A \uparrow}$", r"e $p_{y,A \uparrow}$", r"e $p_{z,S \uparrow}$",
        r"e $d_{z^2 \downarrow}$", r"e $d_{x^2 \downarrow}$", r"e $d_{xy \downarrow}$", r"e $p_{x,S \downarrow}$", r"e $p_{y,S \downarrow}$", r"e $p_{z,A \downarrow}$", r"e $p_{d_{xz} \downarrow}$", r"e $d_{yz \downarrow}$", r"e $p{x,A \downarrow}$", r"e $p_{y,A \downarrow}$", r"e $p_{z,S \downarrow}$",
        r"h $d_{z^2 \uparrow}$", r"h $d_{x^2 \uparrow}$", r"h $d_{xy \uparrow}$", r"h $p_{x,S \uparrow}$", r"h $p_{y,S \uparrow}$", r"h $p_{z,A \uparrow}$", r"h $p_{d_{xz} \uparrow}$", r"h $d_{yz \uparrow}$", r"h $p{x,A \uparrow}$", r"h $p_{y,A \uparrow}$", r"h $p_{z,S \uparrow}$",
        r"h $d_{z^2 \downarrow}$", r"h $d_{x^2 \downarrow}$", r"h $d_{xy \downarrow}$", r"h $p_{x,S \downarrow}$", r"h $p_{y,S \downarrow}$", r"h $p_{z,A \downarrow}$", r"h $p_{d_{xz} \downarrow}$", r"h $d_{yz \downarrow}$", r"h $p{x,A \downarrow}$", r"h $p_{y,A \downarrow}$", r"h $p_{z,S \downarrow}$",
        ]
    n_orbitals = np.shape(orbitals)[0]
    DOS = np.zeros((n_orbitals, np.shape(Es)[0], np.shape(k_values)[0]),dtype='complex')
    eta = 0.001
    if False: 
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
            plt.xticks([0, round(N_points*K_dist/dist_sum), round(N_points*(M_dist+K_M_dist)/dist_sum),N_points], [r"$\Gamma$","M","K",r"$\Gamma$"])
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
            plt.xticks([0, round(N_points*K_dist/dist_sum), round(N_points*(M_dist+K_M_dist)/dist_sum),N_points], [r"$\Gamma$","M","K",r"$\Gamma$"])
        plt.tight_layout()

    if False:
        a = np.expand_dims(np.array(range(22)),1) # electron states
        orbitals = np.concatenate((a,a), axis=1)
        n_orbitals = np.shape(orbitals)[0]
        DOS = np.zeros((np.shape(Es)[0], np.shape(k_values)[0]),dtype='complex')
        eta = 0.001
        for i in range(np.shape(k_values)[0]):
            for j in range(np.shape(Es)[0]):
                for ii in range(n_orbitals):
                    E = Es[j]
                    DOS[j,i] += -1/np.pi*np.sum(cs[i,orbitals[ii,0],:]*np.conj(cs[i,orbitals[ii,1],:])/(E-E_values[i,:]+1j*eta))
            print(i+1)

        plt.figure()
        X, Y = np.meshgrid(range(np.shape(k_values)[0]),Es, indexing='xy')
        plt.pcolormesh(X,Y, np.abs(DOS))
        plt.ylabel("E")
        plt.xlabel("k")
        plt.xticks([0, round(N_points*K_dist/dist_sum), round(N_points*(M_dist+K_M_dist)/dist_sum),N_points], [r"$\Gamma$","M","K",r"$\Gamma$"])
        plt.axhline(color='black', linestyle='--')

        ax = plt.gca()
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        fig.savefig(dirname+"/e_spectrum-{:n}_d2-{:n}_d3-{:n}_d4-{:n}_eta-{:n}.png".format(Delta1*1e3, Delta2*1e3, Delta3*1e3, Delta4*1e3, 1/eta), dpi=200)

    

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
            plt.xticks([0, round(N_points*K_dist/dist_sum), round(N_points*(M_dist+K_M_dist)/dist_sum),N_points], [r"$\Gamma$","M","K",r"$\Gamma$"])
        plt.tight_layout()

    plt.figure()
    plt.plot(BZ_corners()[:,0],BZ_corners()[:,1], "k-")
    plt.plot(k_values[:,0], k_values[:,1], "r-")
    plt.xlabel("$k_x$")
    plt.ylabel("$k_y$")
    plt.scatter(Gamma[0], Gamma[1], marker="o", label="$\Gamma$")
    plt.scatter(M[0], M[1], marker="o", label="M")
    plt.scatter(K[0], K[1], marker="o", label="K")
    plt.legend()
    plt.gca().set_aspect('equal')

    plt.show()
    return


if __name__=="__main__":
    main()