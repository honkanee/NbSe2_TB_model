"""
Implementation of TB Hamiltonian for MoS2 and NbSe2 including SOC and SC

Based on papers:
Cappelluti, E. et al. “Tight-Binding Model and Direct-Gap/indirect-Gap Transition in Single-Layer and Multilayer MoS.” 
Physical review. B, Condensed matter and materials physics 88.7 (2013)

Roldán, R et al. “Momentum Dependence of Spin-Orbit Interaction Effects in Single-Layer and Multi-Layer Transition Metal Dichalcogenides.” 
2d materials 1.3 (2014)

Margalit, Gilad, Erez Berg, and Yuval Oreg. “Theory of Multi-Orbital Topological Superconductivity in Transition Metal Dichalcogenides.” 
Annals of physics 435 (2021)
"""

import numpy as np

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
    """
    Base switch from orginal base 
    [p_{x,t}, p_{y,t}, p_{z,t}, d_{3z^2-r^2}, d_{x^2-y^2}, d_{xy}, d_{xz}, d_{yz}, p_{x,b}, p_{y,b}, p_{z,b}]
    to "even-odd" base
    [d_{3z^2-r^2}, d_{x^2-y^2}, d_{xy}, p_{x,S}, p_{y,S}, p_{z,A}, d_{xz}, d_{yz}, p_{x,A}, p_{y,A}, p_{z,S}]
    """
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

def apply_SC(H, SC_Delta1, SC_Delta2, SC_Delta3, SC_Delta4, Delta_M):
    H_SC = np.zeros((44,44), dtype='complex')

    H_SC[:22, :22] = H
    H_SC[22:, 22:] = -np.conj(H)
    Delta = None

    if Delta_M is None:
        d1 = SC_Delta1
        d2 = SC_Delta2
        d3 = SC_Delta3
        d4 = SC_Delta4
        Delta = np.zeros(np.shape(H), dtype='complex')
        Delta[0,11] = d1
        Delta[1,12] = d2
        Delta[2,13] = d2
        Delta[1,13] = -1j*d3
        Delta[2,12] = 1j*d3
        Delta[0,2] = d4
        Delta[0,1] = 1j*d4
        Delta[11,13] = d4
        Delta[11,12] = -1j*d4
        Delta = Delta - Delta.T
    else:
        Delta = Delta_M

    H_SC[:22, 22:] = Delta
    H_SC[22:, :22] = np.conj(Delta).T
    return H_SC

def Hamilton_MoS2(k, SC_Delta1=0, SC_Delta2=0, SC_Delta3=0, SC_Delta4=0, E0=1.3, Delta_M=None):
    """
    k : two-dimensional momentum vector [k_x, k_y] (np.array)
    SC_Delta1 : singlet pairing in the d_z^2
    SC_Delta2 : singlet pairing in the in-plane orbitals (d_x^2-y^2 and d_xy)
    SC_Delta3 : inter-orbital triplet which pairs the two in-plane orbitals
    SC_Delta4 : another inter-orbital triplet term which notably pairs states of the same spin
    E0 : On-site energy addition to MoS2 model. 1.3 eV used for NbSe2
    Delta_M : Ready-made SC Delta matrix

    returns : 44x44 Hamiltonian matrix (np.array)
    """
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
        HHH += np.identity(np.shape(HHH)[0])*E0 # On-site addition
        HHH = apply_SC(HHH, SC_Delta1, SC_Delta2, SC_Delta3, SC_Delta4, Delta_M)
    return HHH