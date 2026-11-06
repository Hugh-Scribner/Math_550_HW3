import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as spla
import sys

def laplacian_2d(N, L):
    """
    Written by Claude Sonnet
    Sparse 2D Laplacian (5-point stencil) on a square N-by-N grid of nodes
    spanning a domain of side length L (boundary nodes included).

    Node ordering: k = i + j*N (x varies fastest).
    Grid spacing:  h = L / (N - 1).

    No boundary conditions are applied: boundary rows just drop neighbors that
    fall outside the grid. Overwrite those rows later to impose your BCs.

    Returns
    -------
    A : scipy.sparse.csr_matrix of shape (N**2, N**2)
    """
    h = L / (N - 1)

    e = np.ones(N)
    D = spa.diags([e[:-1], -2 * e, e[:-1]], [-1, 0, 1], format="csr") / h**2
    I = spa.identity(N, format="csr")

    A = spa.kron(I, D) + spa.kron(D, I)
    return A


def coloredSpy(A, colormap = "coolwarm", filepath = "Images\\Spy.png"):
    #plt.spy(A, markersize=4, marker='.')
    fig, ax = plt.subplots()
    sc = ax.scatter(A.col, A.row, c=A.data, s=8, cmap=colormap, marker="s")
    ax.set_xlim(-0.5, A.shape[1] - 0.5)
    ax.set_ylim(A.shape[0] - 0.5, -0.5)      # row 0 at the top, like spy
    ax.set_aspect("equal")
    ax.xaxis.tick_top()                       # optional: matches spy's axis placement
    fig.colorbar(sc, ax=ax, label="value")
    plt.savefig(filepath)
    plt.close(fig)
    return 0

def HeatSolver(N, l, num_time_steps, t_max, f, method):
    u = []
    match method:
        case "FE":
            print("Method not yet implemented")
            L = laplacian_2d(N, l)
            I = spa.identity(L.shape[0], format='coo')
            dt = t_max/num_time_steps
            A = I + dt*L
            for i in range(num_time_steps):
                u.append(A @ u[i]) 
        case "BE":
            print("Method not yet implemented")
            L = laplacian_2d(N, l)
            I = spa.identity(L.shape[0], format='coo')
            A = I - dt*L
            for i in range(num_time_steps):
                u.append(spla.solve(A, u[i])) 
        case "CN":
            print("Method not yet implemented")
        case "RK2":
            print("Method not yet implemented")    
        case "RK45"
            print("Method not yet implemented")
        case _:
            print("Invalid method")
            sys.exit()
    return u

def main():
    N, L = 8, 1.0
    A = laplacian_2d(N, L)
    coloredSpy(A)   
    return 0

if __name__ == "__main__":
    main()