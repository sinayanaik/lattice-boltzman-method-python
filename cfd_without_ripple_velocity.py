from platform import freedesktop_os_release
import numpy as np
from matplotlib import pyplot


# distance function

def distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

plot_every = 50

def main():
    Nx = 500 # no of x dimension points
    Ny = 100 # no of y dimension points
    tau = .53 # timescale corresponds to kinematic viscosity
    Nt = 3000 # no of iterations

    # lattice speeds and directions

    NL = 9 # each node has 9 directions

    
    # this convention is arbitrary 0 index starting from middle  
    # cxs : discrete velocity along x direction ,  -1, 0, 1
    # cys : discrete velocity along y direction ,  -1, 0, 1
   
    cxs = np.array([0,0,1,1,1,0,-1,-1,-1])
    cys = np.array([0,1,1,0,-1,-1,-1,0,1])

    weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) 

    # initial condition
    # F : distribution function F(x,y,node,velocity)
    F = np.ones((Ny,Nx,NL)) + .01 * np.random.rand(Ny,Nx,NL) # add slight noise to initial condition
    F[:,:,3] = 2.3 # we want to flow from left --> right so a non-zero velocity in x direction


    # create the obstacle
    # False : empty space
    # True : obstacle present

    cylinder = np.full((Ny,Nx),False)

    # circular obstacle

    for y in range(0,Ny):
        for x in range(0,Nx):
            if distance(x,y,Nx/2,Ny/2) < 13:
                cylinder[y][x] = True

    # main loop
    for iteration in range(0,Nt):
        print(iteration)

        #make the wall absorb fluid
        F[:,-1,[6,7,8]] = F[:,-2,[6,7,8]]
        F[:,0,[2,3,4]] = F[:,1,[2,3,4]]


        # streaming step : every node move t it's corresponding node, 
        # axis 1 = x-axis , axis 0 = y- axis , defined in F  
        for i,cx,cy in zip(range(NL),cxs,cys):
            F[:,:,i] = np.roll(F[:,:,i],cx,axis = 1)
            F[:,:,i] = np.roll(F[:,:,i],cy,axis = 0) 


        # boundary condition

        bndryF = F[cylinder,:]
        bndryF = bndryF[:,[0,5,6,7,8,1,2,3,4]] # invert the discrete velocities


        # fluid variables

        rho = np.sum(F,2) # 2nd index elements
        ux = np.sum(F * cxs ,2) / rho
        uy = np.sum(F * cys ,2) / rho

        # set all velocities inside obstacle to zero

        F[cylinder,:] = bndryF
        ux[cylinder] = 0
        uy[cylinder] = 0

        # collision step
        Feq = np.zeros(F.shape)

        for i,cx,cy,w in zip(range(NL),cxs,cys,weights):
            Feq[:,:,i] =  rho * w * (
                1 + 3 *(cx*ux + cy*uy) + 9 * (cx*ux + cy*uy)**2/2 - 3 * (ux**2 + uy**2)/2
                )
        
        F = F + -(1/tau) * (F-Feq)

        if(iteration % plot_every == 0):
            pyplot.imshow(np.sqrt(ux**2 + uy**2))
            pyplot.pause(0.1)
            pyplot.cla()

















if __name__ == '__main__':
    main()