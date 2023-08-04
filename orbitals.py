"""
Orbitals for real-space projections
"""

import numpy as np
import matplotlib.pyplot as plt

B = np.sqrt((15/(16*np.pi)))

t = np.array([0,0,1.56])
b = -t

def p_x(r):
    orb = np.sqrt((3/(4*np.pi)))*R(np.linalg.norm(r), 1, Z=34)*r[0]/np.linalg.norm(r)
    return orb

def p_y(r):
    orb = np.sqrt((3/(4*np.pi)))*R(np.linalg.norm(r), 1, Z=34)*r[1]/np.linalg.norm(r)
    return orb

def p_z(r):
    orb = np.sqrt(3/(4*np.pi))*R(np.linalg.norm(r), 1, Z=34)*r[2]/np.linalg.norm(r)
    return orb

def p_xS(r):
    orb = (p_x(r+t)+p_x(r+b))/np.sqrt(2)
    return orb

def p_xA(r):
    orb = (p_x(r+t)-p_x(r+b))/np.sqrt(2)
    return orb

def p_yS(r):
    orb = (p_y(r+t)+p_y(r+b))/np.sqrt(2)
    return orb

def p_yA(r):
    orb = (p_y(r+t)-p_y(r+b))/np.sqrt(2)
    return orb

def p_zS(r):
    orb = (p_z(r+t)+p_z(r+b))/np.sqrt(2)
    return orb

def p_zA(r):
    orb = (p_z(r+t)-p_z(r+b))/np.sqrt(2)
    return orb

def d_z2(r):
    r_norm = np.linalg.norm(r)
    orb = B*R(r_norm, 2, Z=41)*(3*r[2]**2-r_norm**2)/r_norm**2
    return orb

def d_xz(r):
    r_norm = np.linalg.norm(r)
    orb = B*R(r_norm, 2, Z=41)*r[0]*r[1]/r_norm**2
    return orb

def d_yz(r):
    r_norm = np.linalg.norm(r)
    orb = B*R(r_norm, 2, Z=41)*r[1]*r[2]/r_norm**2
    return orb

def d_x2(r):
    r_norm = np.linalg.norm(r)
    orb = B*R(r_norm, 2, Z=41)*(r[0]**2-r[1]**2)/r_norm**2
    return orb

def d_xy(r):
    r_norm = np.linalg.norm(r)
    orb = B*R(r_norm, 2, Z=41)*r[0]*r[1]/r_norm**2
    return orb

def Laguerre_5(x,a):
    L2 = x**2/2-(a+2)*x+(a+1)*(a+2)/2
    L3 = -x**3/6+(a+3)*x**2/2-(a+2)*(a+3)/2*x+(a+1)*(a+2)*(a+3)/6
    L4 = (7+a-x)/4*L3-(3+a)/4*L2
    L5 = (9+a-x)/5*L4-(4+a)/5*L3
    return L5

def R(r, l, Z, n=4):
    a = 0.529 # Å (Bohr radius)
    rho = (2*Z/(n*a))*r
    radial_part = -((2*Z/(n*a))**3*(np.math.factorial(n-l-1)/(2*n*np.math.factorial(n+l)**3)))*rho**l*Laguerre_5(rho,2*l+1)*np.exp(-rho/2)
    return radial_part


def main():
    x = np.linspace(-1,1,50)
    y = np.linspace(-1,1,50)
    z = np.linspace(-1,1,50)

    X,Y = np.meshgrid(x,y,indexing='xy')

    orbs = [p_x, p_y, p_z, d_z2, d_xz, d_yz, d_x2, d_xy, p_xS, p_yS, p_zA]
    txt = ["$p_x$", "$p_y$", "$p_z$", "$d_{z^2}$", "$d_{xz}$", "$d_{yz}$", "$d_{x^2-y^2}$", "$d_{xy}$", "$p_{x,S}$", "$p_{y,S}$", "$p_{z,A}$"]
    
    plt.figure()
    for i in range(len(orbs)):
        density = np.zeros((len(x), len(y), len(z)))
        for x_i in range(len(x)):
            for y_i in range(len(y)):
                for z_i in range(len(z)):
                    r = np.array([x[x_i],y[y_i],z[z_i]])
                    orb = orbs[i]
                    density[x_i, y_i, z_i] = orb(r)**2
        plt.subplot(3,4,i+1)
        plt.pcolormesh(X,Y,np.sum(density,axis=2))
        plt.gca().set_aspect('equal')
        plt.xlabel('x [Å]')
        plt.ylabel('y [Å]')
        plt.title(txt[i])
        print(i)
    plt.tight_layout()

    plt.show()

    return


if __name__=="__main__":
    main()