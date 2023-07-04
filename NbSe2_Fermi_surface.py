import numpy as np
import matplotlib.pyplot as plt
from MoS2_SC import Hamilton_MoS2

def color_spins_RGB(up,down):
    G = np.min(np.array([up.T,down.T]),axis=0)
    R = up.T-G
    B = down.T-G

    R = np.expand_dims(R/np.max(R), 2)
    G = np.expand_dims(G/np.max(G), 2)
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
    N = 2000
    kx = np.linspace(-2*np.pi,2*np.pi,N)
    ky = kx

    DOS_e = np.zeros((N,N))
    DOS_e_up = np.zeros((N,N))
    DOS_e_down = np.zeros((N,N))

    F_z2 = np.zeros((N,N), dtype='complex')
    F_x2 = np.zeros((N,N), dtype='complex')
    F_xy = np.zeros((N,N), dtype='complex')

    E0 = 0
    eta = 0.01
    for i in range(N):
        for j in range(N):
            H = Hamilton_MoS2((kx[i], ky[j]))
            E, c = np.linalg.eigh(H)

            DOS_e[i,j] = np.imag(-1/np.pi*np.sum(np.sum(c[:22,:]*np.conj(c[:22,:]),0)/(E0-E+1j*eta)))
            DOS_e_up[i,j] = np.imag(-1/np.pi*np.sum(np.sum(c[:11,:]*np.conj(c[:11,:]),0)/(E0-E+1j*eta)))
            DOS_e_down[i,j] = np.imag(-1/np.pi*np.sum(np.sum(c[11:22,:]*np.conj(c[11:22,:]),0)/(E0-E+1j*eta)))

            F_z2[i,j] = -1/np.pi*np.sum(c[0,:]*np.conj(c[33,:])/(E0-E+1j*eta))#-1/np.pi*np.sum(c[11,:]*np.conj(c[22,:])/(E0-E+1j*eta))
            F_x2[i,j] = -1/np.pi*np.sum(c[1,:]*np.conj(c[34,:])/(E0-E+1j*eta))#-1/np.pi*np.sum(c[12,:]*np.conj(c[23,:])/(E0-E+1j*eta))
            F_xy[i,j] = -1/np.pi*np.sum(c[2,:]*np.conj(c[35,:])/(E0-E+1j*eta))#-1/np.pi*np.sum(c[13,:]*np.conj(c[24,:])/(E0-E+1j*eta))
            
        print(i)

    X,Y = np.meshgrid(kx,ky)
    corners = BZ_corners()

    plt.figure()
    plt.plot(corners[:,0], corners[:,1], 'k')
    plt.pcolormesh(X,Y, DOS_e.T)
    plt.gca().set_aspect('equal')
    plt.xlabel("$k_x$")
    plt.ylabel("$k_y$")
    plt.title("Fermipinta, tilatiheys")

    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    fig.savefig("kuvat/FermiSurf.png", dpi=200)

    plt.figure()
    plt.plot(corners[:,0], corners[:,1], 'k')
    plt.pcolormesh(X,Y, np.abs(DOS_e_up.T), cmap='Blues')
    plt.gca().set_aspect('equal')
    plt.xlabel("$k_x$")
    plt.ylabel("$k_y$")
    plt.title(r"Fermi surface, electron $\uparrow$ density")

    plt.figure()
    plt.imshow(color_spins_RGB(DOS_e_down, DOS_e_up))
    plt.plot(N*(2*np.pi+corners[:,0])/(4*np.pi), N*(2*np.pi+corners[:,1])/(4*np.pi), 'w')
    plt.title("Fermipinta, spinit")

    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    fig.savefig("kuvat/FermiSurf_spins.png", dpi=200)

    plt.figure()
    F = [F_z2,F_x2,F_xy]
    txt = [r"$F_{z^2\uparrow, z^2\downarrow}$", r"$F_{x^2-y^2\uparrow, x^2-y^2\downarrow}$", r"$F_{x-y\uparrow, x-y\downarrow}$"]
    for i in range(3):
        plt.subplot(2,3,i+1)
        plt.pcolormesh(X,Y, np.abs(np.imag(F[i])).T)
        plt.gca().set_aspect('equal')
        plt.ylabel("k_y")
        plt.xlabel("k_x")
        plt.title("Im " + txt[i])
        plt.subplot(2,3,i+4)
        plt.ylabel("k_y")
        plt.xlabel("k_x")
        plt.pcolormesh(X,Y, np.abs(np.real(F[i])).T)
        plt.gca().set_aspect('equal')
        plt.title("Re " + txt[i])
    plt.tight_layout()

    plt.show()

if __name__=="__main__":
    main()