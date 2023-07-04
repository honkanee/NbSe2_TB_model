import numpy as np
import matplotlib.pyplot as plt

SYMMETRY_BLOCKS = True
INCLUDE_SOC = True
INCLUDE_SC = True

sin_phi = np.sqrt(3/7)
cos_phi = np.sqrt(4/7)

# Tight-binding parameters [eV]
#   Crystal fields
Delta0 = -1.512
Delta1 = 0.419 # Not determined
Delta2 = -3.025
Deltap = -1.276
Deltaz = -8.236
#   Intralayer Mo-S
V_pds = -2.619
V_pdp = -1.396
#   Intralayer Mo-Mo
V_dds = -0.933
V_ddp = -0.478
V_ddd = -0.442
#   Intralayer S-S
V_pps = 0.696
V_ppp = 0.278

# SOC
lambda_SO_M = 0.075
lambda_SO_S = 0.052

def base_switch(H):
    a = 1/np.sqrt(2)
    P = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [a, 0, 0, 0, 0, 0, 0, 0, a, 0, 0],
                  [0, a, 0, 0, 0, 0, 0, 0, 0, a, 0],
                  [0, 0, a, 0, 0, 0, 0, 0, 0, 0,-a],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [a, 0, 0, 0, 0, 0, 0, 0,-a, 0, 0],
                  [0, a, 0, 0, 0, 0, 0, 0, 0,-a, 0],
                  [0, 0, a, 0, 0, 0, 0, 0, 0, 0, a]])
    return P@H@np.linalg.inv(P)

def apply_SOC(H):
    HH = np.zeros((np.shape(H)[0]*2, np.shape(H)[1]*2), dtype='complex')
    HH[:11,:11] = H
    HH[11:,11:] = H
    H_SO = np.zeros((np.shape(H)[0]*2, np.shape(H)[1]*2), dtype='complex')

    M_EE_uu = np.array([[0, 0, 0, 0, 0, 0],
                          [0, 0, -1j*lambda_SO_M, 0, 0, 0],
                          [0, 1j*lambda_SO_M, 0, 0, 0, 0],
                          [0, 0, 0, 0, -1j*lambda_SO_S/2, 0],
                          [0, 0, 0, 1j*lambda_SO_S/2, 0, 0],
                          [0, 0, 0, 0, 0, 0]])
    
    M_EE_dd = -M_EE_uu

    M_OO_uu = 1/2*np.array([[0, -1j*lambda_SO_M, 0, 0, 0],
                            [1j*lambda_SO_M, 0, 0, 0, 0],
                            [0, 0, 0, -1j*lambda_SO_S, 0],
                            [0, 0, 1j*lambda_SO_S, 0, 0],
                            [0, 0, 0, 0, 0]])

    M_OO_dd = -M_OO_uu

    M_EO_ud = 1/2*np.array([[-np.sqrt(3)*lambda_SO_M, 1j*np.sqrt(3)*lambda_SO_M, 0, 0, 0],
                            [lambda_SO_M, 1j*lambda_SO_M, 0, 0, 0],
                            [-1j*lambda_SO_M, lambda_SO_M, 0, 0, 0],
                            [0, 0, 0, 0, lambda_SO_S],
                            [0, 0, 0, 0, -1j*lambda_SO_S],
                            [0, 0, -lambda_SO_S, 1j*lambda_SO_S, 0]])
    
    M_OE_du = np.conj(M_EO_ud).T

    M_EO_du = 1/2*np.array([[np.sqrt(3)*lambda_SO_M, 1j*np.sqrt(3)*lambda_SO_M, 0, 0, 0],
                           [-lambda_SO_M, 1j*lambda_SO_M, 0, 0, 0],
                           [-1j*lambda_SO_M, -1j*lambda_SO_M, 0, 0, 0],
                           [0, 0, 0, 0, -lambda_SO_S],
                           [0, 0, 0, 0, -1j*lambda_SO_S],
                           [0, 0, lambda_SO_S, 1j*lambda_SO_S, 0]])

    M_OE_ud = np.conj(M_EO_du).T

    MM_uu = np.zeros(np.shape(H), dtype='complex')
    MM_uu[:6,:6] = M_EE_uu
    MM_uu[6:,6:] = M_OO_uu

    MM_dd = np.zeros(np.shape(H), dtype='complex')
    MM_dd[:6,:6] = M_EE_dd
    MM_dd[6:,6:] = M_OO_dd

    MM_ud = np.zeros(np.shape(H), dtype='complex')
    MM_ud[:6, 6:] = M_EO_ud
    MM_ud[6:, :6] = M_OE_ud

    MM_du = np.zeros(np.shape(H), dtype='complex')
    MM_du[:6, 6:] = M_EO_du
    MM_du[6:, :6] = M_OE_du

    H_SO[:11, :11] = MM_uu
    H_SO[:11, 11:] = MM_ud
    H_SO[11:, :11] = MM_du
    H_SO[11:, 11:] = MM_dd

    return HH + H_SO

def apply_SC(H):
    H_SC = np.zeros((44,44), dtype='complex')

    H += np.identity(np.shape(H)[0])*1.3 # On-site addition
    H_SC[:22, :22] = H
    H_SC[22:, 22:] = -np.conj(H)

    d1 = 0.01
    d2 = 0.01
    d3 = 0.01
    #d1 = 0
    #d2 = 0
    #d3 = 0
    Delta = np.zeros(np.shape(H), dtype='complex')
    Delta[0,11] = d1
    Delta[1,12] = d2
    Delta[2,13] = d2
    Delta[1,13] = 1j*d3
    Delta[2,12] = -1j*d3
    Delta = Delta - Delta.T

    H_SC[:22, 22:] = Delta
    H_SC[22:, :22] = np.conj(Delta).T
    return H_SC

def Hamilton_MoS2(k):
    HHH = np.zeros((11,11),dtype='complex')
    
    # Reduced momentum variables
    xi = k[0]/2
    eta = np.sqrt(3)*k[1]/2

    C1 = 2*np.cos(xi)*np.cos(eta/3)+np.cos(2*eta/3)+1j*(2*np.cos(xi)*np.sin(eta/3)-np.sin(2*eta/3))
    C2 = np.cos(xi)*np.cos(eta/3)-np.cos(2*eta/3)+1j*(np.cos(xi)*np.sin(eta/3)+np.sin(2*eta/3))
    C3 = np.cos(xi)*np.cos(eta/3)+2*np.cos(2*eta/3)+1j*(np.cos(xi)*np.sin(eta/3)-2*np.sin(2*eta/3))
    d1 = np.sin(eta/3)-1j*np.cos(eta/3)
    l1 = np.cos(2*xi)+2*np.cos(xi)*np.cos(eta)
    l2 = np.cos(2*xi)-np.cos(xi)*np.cos(eta)
    l3 = 2*np.cos(2*xi)+np.cos(xi)*np.cos(eta)

    # Intralayer hopping terms
    E1 = 1/2*(-V_pds*(sin_phi**2-1/2*cos_phi**2)+np.sqrt(3)*V_pdp*sin_phi**2)*cos_phi
    E2 = (-V_pds*(sin_phi**2-1/2*cos_phi**2)-np.sqrt(3)*V_pdp*cos_phi**2)*sin_phi
    E3 = 1/4*(np.sqrt(3)/2*V_pds*cos_phi**3+V_pdp*cos_phi*sin_phi**2)
    E4 = 1/2*(np.sqrt(3)/2*V_pds*sin_phi*cos_phi**2-V_pdp*sin_phi*cos_phi**2)
    E5 = -3/4*V_pdp*cos_phi
    E6 = -3/4*V_pdp*sin_phi
    E7 = 1/4*(-np.sqrt(3)*V_pds*cos_phi**2-V_pdp*(1-2*cos_phi**2))*sin_phi
    E8 = 1/2*(-np.sqrt(3)*V_pds*sin_phi**2-V_pdp*(1-2*sin_phi**2))*cos_phi
    E9 = 1/4*V_dds+3/4*V_ddd
    E10 = -np.sqrt(3)/4*(V_dds-V_ddd)
    E11 = 3/4*V_dds+1/4*V_ddd
    E12 = V_ddp
    E13 = V_ddp
    E14 = V_ddd
    E15 = V_pps
    E16 = V_ppp

    H_x_x = Deltap + E15*l3+3*E16*np.cos(xi)*np.cos(eta)
    H_y_y = Deltap + E16*l3+3*E15*np.cos(xi)*np.cos(eta)
    H_z_z = Deltaz + 2*E16*l1
    H_z2_z2 = Delta0 + 2*E9*l1
    H_x2_x2 = Delta2 + E11*l3+3*E12*np.cos(xi)*np.cos(eta)
    H_xy_xy = Delta2 + E12*l3+3*E11*np.cos(xi)*np.cos(eta)
    H_xz_xz = Delta1 + E13*l3+3*E14*np.cos(xi)*np.cos(eta)
    H_yz_yz = Delta1 + E14*l3+3*E13*np.cos(xi)*np.cos(eta)
    H_x_y = -np.sqrt(3)*(E15-E16)*np.sin(xi)*np.sin(eta)
    H_z2_x2 = 2*E10*l2
    H_z2_xy = -2*np.sqrt(3)*E10*np.sin(xi)*np.sin(eta)
    H_x2_xy = np.sqrt(3)*(E11-E12)*np.sin(xi)*np.sin(eta)
    H_xz_yz = np.sqrt(3)*(E14-E13)*np.sin(xi)*np.sin(eta)
    H_z2_x = -2*np.sqrt(3)*E1*np.sin(xi)*d1
    H_z2_y = 2*E1*C2
    H_z2_z = E2*C1
    H_x2_x = -2*np.sqrt(3)*(1/3*E5-E3)*np.sin(xi)*d1
    H_x2_y = -2*E3*C3-2j*E5*np.cos(xi)*d1
    H_x2_z = -2*E4*C2
    H_xy_x = -2/3*E5*C3-6j*E3*np.cos(xi)*d1
    H_xy_y = H_x2_x
    H_xy_z = 2*np.sqrt(3)*E4*np.sin(xi)*d1
    H_xz_x = 2/3*E6*C3+6j*E7*np.cos(xi)*d1
    H_xz_y = 2*np.sqrt(3)*(1/3*E6-E7)*np.sin(xi)*d1
    H_xz_z = -2*np.sqrt(3)*E8*np.sin(xi)*d1
    H_yz_x = H_xz_y
    H_yz_y = 2*E7*C3+2j*E6*np.cos(xi)*d1
    H_yz_z = 2*E8*C2
 
    HH_pb_pb = np.array([[H_x_x, H_x_y, 0], 
                         [np.conj(H_x_y), H_y_y, 0],
                         [0, 0, H_z_z]])
    HH_pt_pt = HH_pb_pb.copy()

    HH_dd = np.array([[H_z2_z2, H_z2_x2, H_z2_xy, 0, 0],
                      [np.conj(H_z2_x2), H_x2_x2, H_x2_xy, 0, 0],
                      [np.conj(H_z2_xy), np.conj(H_x2_xy), H_xy_xy, 0, 0],
                      [0, 0, 0, H_xz_xz, H_xz_yz],
                      [0, 0, 0, np.conj(H_xz_yz), H_yz_yz]])
    
    HH_pt_pb = np.array([[V_ppp, 0, 0],
                        [0, V_ppp, 0],
                        [0, 0, V_pps]])

    HH_d_pt = np.array([[H_z2_x, H_z2_y, H_z2_z],
                        [H_x2_x, H_x2_y, H_x2_z],
                        [H_xy_x, H_xy_y, H_xy_z],
                        [H_xz_x, H_xz_y, H_xz_z],
                        [H_yz_x, H_yz_y, H_yz_z]])

    HH_d_pb = np.array([[H_z2_x, H_z2_y, -H_z2_z],
                        [H_x2_x, H_x2_y, -H_x2_z],
                        [H_xy_x, H_xy_y, -H_xy_z],
                        [-H_xz_x, -H_xz_y, H_xz_z],
                        [-H_yz_x, -H_yz_y, H_yz_x]])

    HHH[:3,:3] = HH_pt_pt
    HHH[:3,3:8] = np.conj(HH_d_pt).T
    HHH[:3,8:] = HH_pt_pb
    HHH[3:8,:3] = HH_d_pt
    HHH[3:8,3:8] = HH_dd
    HHH[3:8,8:] = HH_d_pb
    HHH[8:,:3] = np.conj(HH_pt_pb).T
    HHH[8:,3:8] = np.conj(HH_d_pb).T
    HHH[8:,8:] = HH_pb_pb

    if SYMMETRY_BLOCKS:
        HHH = base_switch(HHH)
    if INCLUDE_SOC:
        HHH = apply_SOC(HHH)
    if INCLUDE_SC:
        HHH =apply_SC(HHH)
    return HHH

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