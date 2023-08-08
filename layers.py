"""
Implementation of three-layer structure NbSe2 (SC)- MoS2 (normal) - NbSe2 (SC)
"""

import numpy as np
import matplotlib.pyplot as plt
import Hamilton_NbSe2
import orbitals
from orbitals_layers_enum import Orb

a = 3.16 # Lattice constant [Å]
R1 = np.array([a,0,0]) # Bravis lattice vectors
R2 = np.array([a/2,np.sqrt(3)/2*a, 0])
R3 = R2-R1

# NbSe2 superconducting Delta parameters
SC_Delta1 = 0.01
SC_Delta2 = 0.01
SC_Delta3 = 0.01
SC_Delta4 = 0.01

factor = 2 # Factor of interlayer Se and S p_z orbitals overlapping compared to the intralayer case 

def Hamliton_layers(k):
    """
    Contructs a 132x132 Hamiltonian matrix at given k
    """

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
    A = Hamilton_NbSe2.apply_SC(A, 0, 0, 0, 0, Delta_M=None)

    H = np.block([
        [H_NbSe2, A, Z],
        [A.T, H_MoS2, A],
        [Z, A.T, H_NbSe2]
        ])

    return H

def Fourier_transform(funs, Rs, N=200, eta=0.01):
    """
    Gives Fourier transforms of functions (funs) from k-space to real space.

    funs : Array of functions of G to be transformed
    Rs : Array of two-dimensional position vectors
    N : Defines number of points used for integration
    eta : eta parameter fo Green's function calculation
    """
    Nx = N
    Ny = int(np.round(N*(4*np.pi/np.sqrt(3))/(8*np.pi/3)))

    K1 = np.array([4*np.pi/3,0])
    K2 = np.array([2*np.pi/3,2*np.pi/np.sqrt(3)])
    
    sums = np.zeros(len(funs), dtype=complex)
    points = 0
    for ky in np.linspace(-K2[1], K2[1], Ny):
        for kx in np.linspace(-K1[0], K1[0],Nx):
            if np.abs(kx) < ((K2[1]-np.abs(ky))/K2[1]*(K1[0]-K2[0])+K2[0]):
                G = np.linalg.inv((1j*eta)*np.identity(132)-Hamliton_layers([kx,ky]))
                for i in range(len(funs)):
                    sums[i] += np.exp(1j*(kx*Rs[i,0]/a+ky*Rs[i,1]/a))*funs[i](G)
                points += 1
    return sums/points

def on_sites_DOS(G):
    rho = 1/(np.pi*1j)*np.imag(np.diagonal(G))
    return rho

def cyclic_boundaries(r):
    """
    Keeps r in a unit cell
    """
    r -= R2/2+R3/2
    b = (r[1]-r[0]/R2[0]*R2[1]) / (-R3[0]*R2[1]/R2[0]+R3[1])
    c = (r[0]-b*R3[0])/R2[0]
    r = np.mod(c,1)*R2+np.mod(b,1)*R3 - R2/2-R3/2 + np.array([0,0,1])*r[2]
    return r

def on_sites_projection_to_real_space(x,y,z, R_a, funs, orbs):
    """
    x : 1D array of x-values
    y : 1D array of y-values
    z : 1D array of z-values
    R_a : Array of atom positions
    funs : Functions of G to be projected and summed
    orbs : Array of corresponding orbital functions

    return : Projection of functions of G into real space, 3D array
    """
    projection = np.zeros((len(x), len(y), len(z)),dtype='complex')
    F_t = Fourier_transform(funs, R_a[:,:2])
    for x_i in range(len(x)):
        for y_i in range(len(y)):
            for z_i in range(len(z)):
                for i in range(len(F_t)):
                    r = np.array([x[x_i], y[y_i], z[z_i]])
                    r = cyclic_boundaries(r)
                    r -= R_a[i]
                    projection[x_i,y_i,z_i] += F_t[i]*orbs[i](r)**2
        print(x_i)

    return projection


def main():

    H = Hamliton_layers([0,0])

    K = np.array([2*np.pi/3,-2*np.pi/np.sqrt(3)])
    M = np.array([np.pi,-np.pi/np.sqrt(3)])

    Gamma = np.zeros(2)

    K_dist = np.linalg.norm(K)
    M_dist = np.linalg.norm(M)
    K_M_dist = np.linalg.norm(K-M)
    dist_sum = K_dist+M_dist+K_M_dist

    N = 1500
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
        H = Hamliton_layers(k)
        E, c = np.linalg.eigh(H)

        E_values[i] = E
        cs[i] = c

        cumulative_H += np.abs(H)


    if True:
        plt.rcParams.update({'font.size': 20})
        titles = ["NbSe$_2$ top", "MoS$_2$, p$_z$ coupling factor {:}".format(factor), "NbSe$_2$ bottom"]
        txt = ["NbSe2_top", "MoS2", "NbSe2_bottom"]
        for layer_i in [1]:
            Es = np.linspace(-0.5,0.5,1500)
            a = np.expand_dims(np.array(range(layer_i*44,layer_i*44+22)),1) # electron states
            orbits = np.concatenate((a,a), axis=1)
            n_orbitals = np.shape(orbits)[0]
            DOS = np.zeros((n_orbitals, np.shape(Es)[0], np.shape(k_values)[0]),dtype='complex')
            eta = 0.001
            for ii in range(n_orbitals):
                for i in range(np.shape(k_values)[0]):
                    for j in range(np.shape(Es)[0]):
                        E = Es[j]
                        DOS[ii,j,i] = -1/np.pi*np.sum(cs[i,orbits[ii,0],:]*np.conj(cs[i,orbits[ii,1],:])/(E-E_values[i,:]+1j*eta))
                print(ii+1)

            plt.figure()
            X, Y = np.meshgrid(range(np.shape(k_values)[0]),Es)
            plt.subplot(1,2,1)
            for i in range(n_orbitals):
                plt.pcolormesh(X,Y, np.abs(np.imag(np.sum(DOS,0))))
            plt.colorbar()
            plt.ylabel("E")
            plt.xlabel("k")
            plt.xticks([0, round(N*K_dist/dist_sum), round(N*(M_dist+K_M_dist)/dist_sum),N], [r"$\Gamma$","M","K",r"$\Gamma$"])
            plt.title(titles[layer_i])
            plt.subplot(1,2,2)
            l = int(round(3/8*len(Es)))
            m = int(round(5/8*len(Es)))
            y = np.linspace(-0.5,0.5,m-l)
            X, Y = np.meshgrid(range(np.shape(k_values)[0]),y)
            for i in range(n_orbitals):
                A = np.abs(np.imag(np.sum(DOS,0)))[l:m,:]
                plt.pcolormesh(X,Y, A)
            plt.colorbar()
            plt.ylabel("E")
            plt.xlabel("k")
            plt.xticks([0, round(N*K_dist/dist_sum), round(N*(M_dist+K_M_dist)/dist_sum),N], [r"$\Gamma$","M","K",r"$\Gamma$"])
            plt.title(titles[layer_i])
            plt.gcf().set_size_inches(16,8)
            plt.savefig("./kuvat/layers/bond_factor_{0}_".format(factor)+txt[layer_i], dpi=2000)

    # Real space projections

    N = 100
    x = np.linspace(-3,3,N)
    y = np.linspace(-3,3,N)
    z = np.array([0])

    R_a = np.array([[0,-0.91,0], [0,-0.91,0], [0,-0.91,0], 
                    R2+[0,-0.91,0], R2+[0,-0.91,0], R2+[0,-0.91,0],
                    R3+[0,-0.91,0], R3+[0,-0.91,0], R3+[0,-0.91,0],
                    [0,0.91,0], [0,0.91,0], [0,0.91,0],
                    [0,0.91,0]-R2, [0,0.91,0]-R2, [0,0.91,0]-R2,
                    [0,0.91,0]-R3, [0,0.91,0]-R3, [0,0.91,0]-R3])
    funs = [lambda G : on_sites_DOS(G)[0], lambda G : on_sites_DOS(G)[1], lambda G : on_sites_DOS(G)[2],
            lambda G : on_sites_DOS(G)[0], lambda G : on_sites_DOS(G)[1], lambda G : on_sites_DOS(G)[2],
            lambda G : on_sites_DOS(G)[0], lambda G : on_sites_DOS(G)[1], lambda G : on_sites_DOS(G)[2],
            lambda G : on_sites_DOS(G)[3], lambda G : on_sites_DOS(G)[4], lambda G : on_sites_DOS(G)[5],
            lambda G : on_sites_DOS(G)[3], lambda G : on_sites_DOS(G)[4], lambda G : on_sites_DOS(G)[5],
            lambda G : on_sites_DOS(G)[3], lambda G : on_sites_DOS(G)[4], lambda G : on_sites_DOS(G)[5]]
    orbs = [orbitals.d_z2, orbitals.d_x2, orbitals.d_xy,
            orbitals.d_z2, orbitals.d_x2, orbitals.d_xy,
            orbitals.d_z2, orbitals.d_x2, orbitals.d_xy,
            orbitals.p_xS, orbitals.p_yS, orbitals.p_zA,
            orbitals.p_xS, orbitals.p_yS, orbitals.p_zA,
            orbitals.p_xS, orbitals.p_yS, orbitals.p_zA]

    rs_proj = on_sites_projection_to_real_space(x,y,z, R_a, funs, orbs)

    plt.figure()
    X,Y = np.meshgrid(x,y)
    plt.pcolormesh(X,Y, abs(np.sum(rs_proj,axis=2)).T)
    plt.gca().set_aspect('equal')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()

    plt.scatter(np.array([0]),np.array([-0.91]), color="r", marker="x", label="Nb")
    plt.scatter(np.array([0]),np.array([0.91]), color ="g", marker="x", label="Se$_2$")
    plt.legend()
    plt.title("Top NbSe$_2$, electron on-sites, z=0")

    r = np.zeros((N,N))
    for x_i in range(N):
        for y_i in range(N):
            r[x_i,y_i] = np.linalg.norm(cyclic_boundaries(np.array([x[x_i],y[y_i], 0])))
    plt.figure()
    plt.pcolormesh(X,Y, r.T)
    plt.gca().set_aspect('equal')

    #plt.show()


    return

if __name__=="__main__":
    main()