import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa
import scipy.linalg as spla
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

def heatSolver(N, l, f, u_0, t_max, t_steps=None, method="FE"):
    '''
    This function solves the heat equation using the method of finite differences and a users choice of method

    inputs
    ------
    N : number of internal nodes in one direction in spatial variables
    l : length of one spatial edge
    u0 : initial condition
    f : forcing function
    t_max : time you want to solve the heat equation till
    t_steps : the number of steps you want to take to get to t_max
    method : Short string that describes the method you want to use for time stepping

    outputs
    -------
    sol : 4 item list containing X grid, Y grid, time of each step, and a list of snap shots of u 
    '''
    u = []
    N_trunc= N-2

    # Build mesh and Right hand side
    dx = l/(N-1)
    x = np.linspace(0, l, num = N, endpoint = True)
    y = np.linspace(0, l, num = N, endpoint = True)
    x = x[1:-1] # Truncate off boundaries 
    y = y[1:-1] 
    xx, yy = np.meshgrid(x,y, indexing = 'ij')
    xx_vec, yy_vec = xx.ravel(), yy.ravel() # get grids into a convenient shape for multiple calcs of f

    b = u_0(xx,yy,0)
    
    # Apply BCs
    b[0, :] -= 0 # Left hand BC
    b[-1, :] -= 0 # right hand BC
    b[:, 0] -= 0 # top BC
    b[:, -1] -= 0 # right BC

    u.append(b)

    # run time stepper
    match method:
        case "FE":
            print("Building FE system")
            if t_steps is None:   # enforce CFL condition on time stepping
                t_steps = int(np.ceil(4.0 * t_max / dx**2))
            L = laplacian_2d(N-2, l-2.0*dx) # take off 2 in each dimension 
            I = spa.identity(L.shape[0], format='coo')
            dt = t_max/t_steps
            A = I + dt*L
            A = A.tocsr()
            u_temp = u[0].ravel()
            print("Solving FE system")
            for i in range(t_steps):
                #print(f"solving for t={(i+1)*dt}")
                u_temp = A @ u_temp + dt * f(xx_vec,yy_vec,i*dt) # We use tn for FE
                u.append(u_temp.reshape(N_trunc,N_trunc)) 
        case "BE":
            print("Building BE system")
            if t_steps is None:   # enforce CFL condition on time stepping
                t_steps = int(np.ceil(4.0 * t_max / dx**2)) # technically not required for BE, but we'll reuse backward euler CFL
            L = laplacian_2d(N-2, l-2.0*dx) # take off 2 in each dimension 
            I = spa.identity(L.shape[0], format='coo')
            dt = t_max/t_steps
            A = I - dt*L
            A = A.toarray()
            A_chol = spla.cho_factor(A) # Claude recommended a banded cholesky solve instead - is that important for this homework?
            u_temp = u[0].ravel()
            print("Solving FE system")
            for i in range(t_steps):
                print(f"solving for t={(i+1)*dt}")
                z = u_temp + dt * f(xx_vec,yy_vec,(i+1)*dt) # Apply the forcing update to the time step, use tn+1 for BE
                u_temp = spla.cho_solve(A_chol,z)
                u.append(u_temp.reshape(N_trunc,N_trunc)) 
        case "CN":
            print("Method not yet implemented")
        case "RK2":
            print("Method not yet implemented")    
        case "RK45":
            print("Method not yet implemented")
        case _:
            print("Invalid method")
            sys.exit()
    sol = [xx,yy,u]
    return sol

def main():
    colormap = "inferno"
    N, L = 128, 2*np.pi
    f = lambda x,y,t: 2.0 * np.sin(x) * np.sin(y) * (np.cos(10.0*t) - 5.0 * np.sin(10.0 * t))
    u0 = lambda x,y,t: np.sin(x) * np.sin(y)
    u_exact = lambda x,y,t : np.cos(10.0 * t) * np.sin(x) * np.sin(y)
    t_max = 2.0*np.pi
    t_steps = 10

    # Get solutions
    #sol_FE = heatSolver(N, L, f, u0, t_max, t_steps, method="FE")
    sol_FE = heatSolver(N, L, f, u0, t_max, t_steps, method="BE")
    sol_exact = u_exact(sol_FE[0], sol_FE[1], t_max)
    err = sol_FE[2][-1] - sol_exact

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(sol_FE[0], sol_FE[1], err, cmap=colormap)
    fig.colorbar(surf, shrink=0.6, aspect=12, label="z")
    plt.show()

    return 0

if __name__ == "__main__":
    main()