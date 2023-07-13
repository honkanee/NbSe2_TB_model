import numpy as np
import matplotlib.pyplot as plt
from Hamilton_NbSe2 import Hamilton_MoS2
import os

N = 3000
E0 = 0
eta = 0.001

SC_Delta1 = 0.01 #  singlet pairing in the d_z^2
SC_Delta2 = 0.01 # singlet pairing in the in-plane orbitals (d_x^2−y^2 and d_xy)
SC_Delta3 = 0.01 # inter-orbital triplet which pairs the two in-plane orbitals

def color_spins_RGB(up,down):
    G = np.min(np.array([up.T,down.T]),axis=0)
    R = up.T-G
    B = down.T-G

    G = np.expand_dims(G/np.max([np.max(up),np.max(down)]), 2)
    R = np.expand_dims(R/np.max(R), 2)
    B = np.expand_dims(B/np.max(B), 2)
    RGB = np.concatenate((R,G,B), 2)
    return RGB

def BZ_corners():
    K1 = np.array([4*np.pi/3,0])
    K2 = np.array([2*np.pi/3,2*np.pi/np.sqrt(3)])
    K3 = np.array([-2*np.pi/3,2*np.pi/np.sqrt(3)])
    K4 = np.array([-4*np.pi/3,0])
    K5 = np.array([-2*np.pi/3,-2*np.pi/np.sqrt(3)])
    K6 = np.array([2*np.pi/3,-2*np.pi/np.sqrt(3)])
    corners = np.array([K1, K2, K3, K4, K5, K6, K1])
    return corners

def main():

    print(" N = {:} \n eta = {:} \n Delta1 = {:} \n Delta2 = {:} \n Delta3 = {:}".format(N, eta, SC_Delta1, SC_Delta2, SC_Delta3))

    gaps = np.array([0, -0.037, 0.037, -0.055, 0.55])
    for E0 in gaps:
        print(E0)
        kx = np.linspace(-2*np.pi,2*np.pi,N)
        ky = kx

        DOS_e = np.zeros((N,N))
        DOS_e_up = np.zeros((N,N))
        DOS_e_down = np.zeros((N,N))

        F_z2 = np.zeros((N,N), dtype='complex')
        F_x2 = np.zeros((N,N), dtype='complex')
        F_xy = np.zeros((N,N), dtype='complex')

        F_z2_uu = np.zeros((N,N), dtype='complex')
        F_x2_uu = np.zeros((N,N), dtype='complex')
        F_xy_uu = np.zeros((N,N), dtype='complex')

        for i in range(N):
            for j in range(N):
                H = Hamilton_MoS2((kx[i], ky[j]), SC_Delta1, SC_Delta2, SC_Delta3)
                E, c = np.linalg.eigh(H)

                DOS_e[i,j] = np.imag(-1/np.pi*np.sum(np.sum(c[:22,:]*np.conj(c[:22,:]),0)/(E0-E+1j*eta)))
                DOS_e_up[i,j] = np.imag(-1/np.pi*np.sum(np.sum(c[:11,:]*np.conj(c[:11,:]),0)/(E0-E+1j*eta)))
                DOS_e_down[i,j] = np.imag(-1/np.pi*np.sum(np.sum(c[11:22,:]*np.conj(c[11:22,:]),0)/(E0-E+1j*eta)))

                F_z2[i,j] = -1/np.pi*np.sum(c[0,:]*np.conj(c[33,:])/(E0-E+1j*eta))#-1/np.pi*np.sum(c[11,:]*np.conj(c[22,:])/(E0-E+1j*eta))
                F_x2[i,j] = -1/np.pi*np.sum(c[1,:]*np.conj(c[34,:])/(E0-E+1j*eta))#-1/np.pi*np.sum(c[12,:]*np.conj(c[23,:])/(E0-E+1j*eta))
                F_xy[i,j] = -1/np.pi*np.sum(c[2,:]*np.conj(c[35,:])/(E0-E+1j*eta))#-1/np.pi*np.sum(c[13,:]*np.conj(c[24,:])/(E0-E+1j*eta))

                F_z2_uu[i,j] = -1/np.pi*np.sum(c[0,:]*np.conj(c[33,:])/(E0-E+1j*eta))
                F_x2_uu[i,j] = -1/np.pi*np.sum(c[1,:]*np.conj(c[34,:])/(E0-E+1j*eta))
                F_xy_uu[i,j] = -1/np.pi*np.sum(c[2,:]*np.conj(c[35,:])/(E0-E+1j*eta))
                
            if i%100 == 0:
                print('%.0f'% (100*i/N))

        X,Y = np.meshgrid(kx,ky)
        corners = BZ_corners()

        dirname = "./kuvat/surf_E{0}meV".format(np.round(E0*1000))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        plt.figure()
        plt.plot(corners[:,0], corners[:,1], 'k')
        plt.pcolormesh(X,Y, DOS_e.T)
        plt.gca().set_aspect('equal')
        plt.xlabel("$k_x$")
        plt.ylabel("$k_y$")
        plt.title("Fermipinta, tilatiheys")

        textstr = '\n'.join((
        r'$E=%.0f$ meV' % (E0 * 1000),
        r'$\Delta_1=%.0f$ meV' % (SC_Delta1*1000, ),
        r'$\Delta_2=%.0f$ meV' % (SC_Delta2*1000, ),
        r'$\Delta_3=%.0f$ meV' % (SC_Delta3*1000)))

        ax = plt.gca()
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        
        fig = plt.gcf()
        fig.set_size_inches(8, 8)
        fig.savefig(dirname+"/surf_dos_d1-{:n}_d2-{:n}_d3-{:n}_eta-{:n}.png".format(SC_Delta1*1e3, SC_Delta2*1e3, SC_Delta3*1e3, 1/eta), dpi=200)

        plt.figure()
        plt.plot(corners[:,0], corners[:,1], 'k')
        plt.pcolormesh(X,Y, np.abs(DOS_e_up.T), cmap='Blues')
        plt.gca().set_aspect('equal')
        plt.colorbar()
        plt.xlabel("$k_x$")
        plt.ylabel("$k_y$")
        plt.title(r"Fermi surface, electron $\uparrow$ density")

        plt.figure()
        plt.imshow(color_spins_RGB(DOS_e_down, DOS_e_up))
        plt.plot(N*(2*np.pi+corners[:,0])/(4*np.pi), N*(2*np.pi+corners[:,1])/(4*np.pi), 'w')
        plt.title("Fermipinta, spinit")

        fig = plt.gcf()
        fig.set_size_inches(8, 8)
        plt.gca().text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
        fig.savefig(dirname+"/surf_spins_d1-{:n}_d2-{:n}_d3-{:n}_eta-{:n}.png".format(SC_Delta1*1e3, SC_Delta2*1e3, SC_Delta3*1e3, 1/eta), dpi=200)


        fig = plt.figure()
        F = [F_z2_uu,F_x2_uu,F_xy_uu]
        max_F_imag = np.max(np.imag(F))
        max_F_real = np.max(np.real(F))
        txt = [r"$F_{z^2\uparrow, z^2\uparrow}$", r"$F_{x^2-y^2\uparrow, x^2-y^2\uparrow}$", r"$F_{xy\uparrow, xy\uparrow}$"]
        for i in range(3):
            plt.subplot(2,3,i+1)
            plt.pcolormesh(X,Y, np.abs(np.imag(F[i])).T, vmin=0, vmax=max_F_imag)
            plt.gca().set_aspect('equal')
            plt.ylabel("k_y")
            plt.xlabel("k_x")
            plt.title("Im " + txt[i])
            plt.subplot(2,3,i+4)
            plt.ylabel("k_y")
            plt.xlabel("k_x")
            plt.pcolormesh(X,Y, np.abs(np.real(F[i])).T, vmin=0, vmax=max_F_real)
            plt.gca().set_aspect('equal')
            plt.title("Re " + txt[i])
        fig.set_size_inches(12, 8)
        fig.text(0.062, 0.95, textstr, fontsize=14,
        verticalalignment='top', bbox=props)
        plt.tight_layout()
        fig.savefig(dirname+"/F_triplet_elements_d1-{:n}_d2-{:n}_d3-{:n}_eta-{:n}.png".format(SC_Delta1*1e3, SC_Delta2*1e3, SC_Delta3*1e3, 1/eta), dpi=200)

        fig = plt.figure()
        F = [F_z2,F_x2,F_xy]
        max_F_imag = np.max(np.imag(F))
        max_F_real = np.max(np.real(F))
        txt = [r"$F_{z^2\uparrow, z^2\downarrow}$", r"$F_{x^2-y^2\uparrow, x^2-y^2\downarrow}$", r"$F_{xy\uparrow, xy\downarrow}$"]
        for i in range(3):
            plt.subplot(2,3,i+1)
            plt.pcolormesh(X,Y, np.abs(np.imag(F[i])).T, vmin=0, vmax=max_F_imag)
            plt.gca().set_aspect('equal')
            plt.ylabel("k_y")
            plt.xlabel("k_x")
            plt.title("Im " + txt[i])
            plt.subplot(2,3,i+4)
            plt.ylabel("k_y")
            plt.xlabel("k_x")
            plt.pcolormesh(X,Y, np.abs(np.real(F[i])).T, vmin=0, vmax=max_F_real)
            plt.gca().set_aspect('equal')
            plt.title("Re " + txt[i])
        fig.set_size_inches(12, 8)
        fig.text(0.062, 0.95, textstr, fontsize=14,
        verticalalignment='top', bbox=props)
        plt.tight_layout()
        fig.savefig(dirname+"/F_singlet_elements_d1-{:n}_d2-{:n}_d3-{:n}_eta-{:n}.png".format(SC_Delta1*1e3, SC_Delta2*1e3, SC_Delta3*1e3, 1/eta), dpi=200)

        plt.close('all')

    plt.show()

if __name__=="__main__":
    main()