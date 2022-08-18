import numpy as np
from matplotlib import pyplot as plt


# distance function

def distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


plot_every = 50


def main():
    Nx = 500  # no of x dimension points
    Ny = 100  # no of y dimension points
    tau = .53  # timescale corresponds to kinematic viscosity
    iter_count = 3000  # no of iterations

    # lattice speeds and directions

    Node_dirn = 9

    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])

    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    def shape_1():
        F_1 = np.ones((Ny, Nx, Node_dirn)) + .01 * \
            np.random.rand(Ny, Nx, Node_dirn)
        F_1[:, :, 3] = 2.3
        space_1 = np.full((Ny, Nx), False)

        for y in range(0, Ny):
            for x in range(0, Nx):
                if distance(x, y, Nx/2, Ny/2) < 13:
                    space_1[y][x] = True

        # main loop
      

            # make the wall absorb fluid
            F_1[:, -1, [6, 7, 8]] = F_1[:, -2, [6, 7, 8]]
            F_1[:, 0, [2, 3, 4]] = F_1[:, 1, [2, 3, 4]]

            # streaming step : every node move t it's corresponding node,
            # axis 1 = x-axis , axis 0 = y- axis , defined in F
            for i, cx, cy in zip(range(Node_dirn), cxs, cys):
                F_1[:, :, i] = np.roll(F_1[:, :, i], cx, axis=1)
                F_1[:, :, i] = np.roll(F_1[:, :, i], cy, axis=0)

            # boundary condition

            bndryF_1 = F_1[space_1, :]
            # invert the discrete velocities
            bndryF_1 = bndryF_1[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

            # fluid variables

            rho_1 = np.sum(F_1, 2)  # 2nd index elements
            ux_1 = np.sum(F_1 * cxs, 2) / rho_1
            uy_1 = np.sum(F_1 * cys, 2) / rho_1

            # set all velocities inside obstacle to zero

            F_1[space_1, :] = bndryF_1
            ux_1[space_1] = 0
            uy_1[space_1] = 0

            # collision step
            Feq_1 = np.zeros(F_1.shape)

            for i, cx, cy, w in zip(range(Node_dirn), cxs, cys, weights):
                Feq_1[:, :, i] = rho_1 * w * (
                    1 + 3 * (cx*ux_1 + cy*uy_1) + 9 * (cx*ux_1 + cy*uy_1)**2 /
                    2 - 3 * (ux_1**2 + uy_1**2)/2
                )

            F_1 = F_1 + -(1/tau) * (F_1-Feq_1)

    def shape_2():
        F_2 = np.ones((Ny, Nx, Node_dirn)) + .01 * np.random.rand(Ny,
                                                                  Nx, Node_dirn)  # add slight noise to initial condition
    # we want to flow from left --> right so a non-zero velocity in x direction
        F_2[:, :, 3] = 2.3

        # create the obstacle
        # False : empty space
        # True : obstacle present

        space_2 = np.full((Ny, Nx), False)

        # circular obstacle

        for y in range(0, Ny):
            for x in range(0, Nx):
                if distance(x, y, Nx/2, Ny/2) < 13:
                    space_2[y][x] = True

        # main loop
        for iteration in range(0, iter_count):
            print(iteration)

            # streaming step : every node move t it's corresponding node,
            # axis 1 = x-axis , axis 0 = y- axis , defined in F
            for i, cx, cy in zip(range(Node_dirn), cxs, cys):
                F_2[:, :, i] = np.roll(F_2[:, :, i], cx, axis=1)
                F_2[:, :, i] = np.roll(F_2[:, :, i], cy, axis=0)

            # boundary condition

            bndryF_2 = F_2[space_2, :]
            # invert the discrete velocities
            bndryF_2 = bndryF_2[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

            # fluid variables

            rho_2 = np.sum(F_2, 2)  # 2nd index elements
            ux_2 = np.sum(F_2 * cxs, 2) / rho_2
            uy_2 = np.sum(F_2 * cys, 2) / rho_2

            # set all velocities inside obstacle to zero

            F_2[space_2, :] = bndryF_2
            ux_2[space_2] = 0
            uy_2[space_2] = 0

            # collision step
            Feq_2 = np.zeros(F_2.shape)

            for i, cx, cy, w in zip(range(Node_dirn), cxs, cys, weights):
                Feq_2[:, :, i] = rho_2 * w * (
                    1 + 3 * (cx*ux_2 + cy*uy_2) + 9 * (cx*ux_2 + cy*uy_2)**2 /
                    2 - 3 * (ux_2**2 + uy_2**2)/2
                )

            F_2 = F_2 + -(1/tau) * (F_2-Feq_2)

    shape_1()
    shape_2()
    for iteration in range(0, iter_count):
        print(iteration)
        if(iteration % plot_every == 0):
                u1 = np.sum(F_1 * cxs, 2) / np.sum(F_1, 2)
                u2 = np.sum(F_2 * cxs, 2) / np.sum(F_2, 2)
                # plot both u1 and u2 in the same plot
                plt.plot(u1[:, Nx/2], 'r', label='u1')
                plt.plot(u2[:, Nx/2], 'b', label='u2')
                plt.legend()
                plt.show()
                plt.clf()


if __name__ == '__main__':
    main()
