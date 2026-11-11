import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as spla
import sys
import io
import contextlib
from matplotlib.lines import Line2D
from scipy.integrate import RK45

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
            A_lu = spla.splu(A) # Claude recommended a banded cholesky solve instead - is that important for this homework?
            u_temp = u[0].ravel()
            print("Solving FE system")
            for i in range(t_steps):
                z = u_temp + dt * f(xx_vec,yy_vec,(i+1)*dt) # Apply the forcing update to the time step, use tn+1 for BE
                u_temp = A_lu.solve(z)
                u.append(u_temp.reshape(N_trunc,N_trunc)) 
        case "CN":
            # CN solver
            # Set up required values
            print("Building CN system")
            if t_steps is None:   # enforce CFL condition on time stepping
                t_steps = int(np.ceil(t_max/.1))
            L = laplacian_2d(N-2, l-2.0*dx) # take off 2 in each dimension 
            I = spa.identity(L.shape[0], format='coo')
            dt = t_max/t_steps
        
            # build matricies
            A_f = (I + dt/2*L).tocsr()
            A_b = (I - dt/2*L)
            A_b_lu = spla.splu(A_b)
        
            # initialize RHS
            u_temp = u[0].ravel()
        
            print("Solving CN system")
            for i in range(t_steps):
                z = A_f @ u_temp + dt/2.0 * (f(xx_vec,yy_vec,i*dt) + f(xx_vec,yy_vec, (i+1)*dt))
                u_temp = A_b_lu.solve(z)
                u.append(u_temp.reshape(N_trunc,N_trunc))
        case "RK2":
            print("Building RK2 system")
            if t_steps is None:   # enforce CFL condition on time stepping
                t_steps = int(np.ceil(4.0 * t_max / dx**2)) # technically not required for BE, but we'll reuse backward euler CFL
            L = laplacian_2d(N-2, l-2.0*dx) # take off 2 in each dimension 
            dt = t_max/t_steps
            u_temp = u[0].ravel()
            print("Solving RK2 system")
            for i in range(t_steps):
                k1 = dt/2.0 * (L @ u_temp + f(xx_vec, yy_vec, i*dt))
                k2 = dt * (L @ (u_temp+k1) + f(xx_vec, yy_vec, (i + 0.5)*dt))
                u_temp = u_temp + k2
                u.append(u_temp.reshape(N_trunc,N_trunc))     
        case "BDF2":
            print("Building BDF system")
            if t_steps is None:   # enforce CFL condition on time stepping
                t_steps = int(np.ceil(4.0 * t_max / dx**2)) # technically not required for BE, but we'll reuse backward euler CFL
            L = laplacian_2d(N-2, l-2.0*dx) # take off 2 in each dimension 
            I = spa.identity(L.shape[0], format='csc')
            dt = t_max/t_steps 
            A_BE = I - dt*L # A for BE

            # Cholesky factorization for BFD
            A_BDF = (I - 2.0/3.0 * dt * L)
            A_BDF_lu = spla.splu(A_BDF)

            # Calc BE step, cholesky doesn't pay off here
            u_temp0 = u[0].ravel()
            u_temp1 = spla.spsolve(A_BE, u_temp0 + dt * f(xx_vec,yy_vec,dt)) # Use a FE step to get first multistep
            u.append(u_temp1.reshape(N_trunc,N_trunc))
           
            print("Solving BDF system")
            for i in range(1,t_steps):
                z = 4.0 / 3.0 * u_temp1 - 1.0/3.0 * u_temp0 + 2.0/3.0 * dt * f(xx_vec,yy_vec,(i+1)*dt) # Apply the forcing update to the time step, use tn+1 for BE
                u_temp0 = u_temp1
                u_temp1 = A_BDF_lu.solve(z)
                u.append(u_temp1.reshape(N_trunc,N_trunc)) 
        case "RK45":
            L = laplacian_2d(N-2, l-2.0*dx)
            ode = lambda t,c : L @ c + f(xx_vec, yy_vec, t)
            u_temp = u[0].ravel() 
            solver = RK45(ode, 0, u_temp, t_max, rtol=1e-5, atol=1e-8)
            while(solver.t < t_max):
                print(solver.t)
                solver.step()
                u.append((solver.y).reshape(N_trunc,N_trunc))
        case _:
            print("Invalid method")
            sys.exit()
            
    sol = [xx,yy,u]
    return sol

def rel_error(u_exact, u_approx):
    """
    This code was modified by claude
    Relative Frobenius-norm error. Returns np.nan if the approximation
    contains nan/inf (i.e. the method blew up).
    """
    if not np.all(np.isfinite(u_approx)):
        return np.nan
    with np.errstate(all='ignore'):
        err = np.linalg.norm(u_exact - u_approx, ord='fro') / np.linalg.norm(u_exact, ord='fro')
    return err

def run(method, N, steps, exact_sol, f, u0, t_max, l, diverge_tol=1e3):
    """
    This code was written by claude
    Run one heatSolver simulation and return its relative error at t_max.
    Returns np.nan if the run diverged (nan/inf or error > diverge_tol) or raised an exception.
    """
    try:
        # silence heatSolver's prints (RK45 prints every step) and overflow warnings
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all='ignore'):
            xx, yy, u = heatSolver(N, l, f, u0, t_max, steps, method=method)
            err = rel_error(exact_sol(xx, yy, t_max), u[-1])
    except Exception as e:
        print(f"    {method} N={N} steps={steps} failed: {e}")
        err = np.nan
    if not np.isfinite(err) or err > diverge_tol:
        err = np.nan
    return err

def convTest(exact_sol, f, u0, t_max, l, meshes=(64, 128),
             step_counts=(15, 50, 100, 500, 1000, 5000, 10300, 25000),
             filepath='Images\\convergence_plot.png', diverge_tol=1e3,
             highlight_methods=("FE", "BE", "RK2")):
    """
    This code was largely generated by claude from a script I used extensively in other homeworks.
    Time-step convergence test for every method in heatSolver, on one plot.

    inputs
    ------
    exact_sol   : exact solution, exact_sol(x, y, t)
    f, u0       : forcing function and initial condition (as in heatSolver)
    t_max, l    : final time and domain edge length
    meshes      : (coarse N, fine N). Coarse is drawn solid, fine is dashed.
    step_counts : numbers of time steps to try for the fixed-step methods
    filepath    : where to save the figure
    diverge_tol : a run whose relative error is nan/inf or exceeds this is
                  treated as diverged and left off the plot (the curve breaks there)
    highlight_methods : methods whose curves on the fine mesh are drawn bold and on top

    outputs
    -------
    results : dict {(method, N): array of errors (nan = diverged)}; RK45 has a single float
    """
    fixed_methods = ["FE", "BE", "CN", "RK2", "BDF2"]
    all_methods = fixed_methods + ["RK45"]
    colors = {m: plt.cm.tab10(i) for i, m in enumerate(all_methods)}
    linestyles = {meshes[0]: '-', meshes[1]: '--'}
    step_counts = np.asarray(step_counts)

    results = {}
    fig, ax = plt.subplots(figsize=(9, 6))

    for N in meshes:
        # Fixed-step methods: sweep over the number of time steps
        for m in fixed_methods:
            errs = np.full(len(step_counts), np.nan)
            for j, steps in enumerate(step_counts):
                errs[j] = run(m, N, int(steps), exact_sol, f, u0, t_max, l, diverge_tol)
                status = "diverged" if np.isnan(errs[j]) else f"error = {errs[j]:.3e}"
                print(f"{m:5s} N={N:4d} steps={steps:6d}: {status}")
            results[(m, N)] = errs
            bold = (N == meshes[1]) and (m in highlight_methods)
            ax.loglog(step_counts, errs, linestyle=linestyles[N], color=colors[m],
                      marker='o', markersize=8 if bold else 4,
                      linewidth=4.5 if bold else 1.5,
                      alpha=1.0 if bold else 0.7,
                      zorder=3 if bold else 2)

        # RK45: adaptive, so run once and draw a flat line across the plot
        err45 = run("RK45", N, None, exact_sol, f, u0, t_max, l, diverge_tol)
        print(f"RK45  N={N:4d} (adaptive): " + ("diverged" if np.isnan(err45) else f"error = {err45:.3e}"))
        results[("RK45", N)] = err45
        if np.isfinite(err45):
            ax.loglog([step_counts[0], step_counts[-1]], [err45, err45],
                      linestyle=linestyles[N], color=colors["RK45"], linewidth=1.5,
                      alpha=0.7, zorder=2)

    # Legend: color -> method, line style -> mesh
    handles = [Line2D([0], [0], color=colors[m], linewidth=2, label=m) for m in all_methods]
    handles += [Line2D([0], [0], color='black', linestyle=linestyles[N], linewidth=2,
                       label=f"{N} x {N} grid") for N in meshes]
    ax.legend(handles=handles, loc='best')
    ax.set_xlabel("Number of time steps")
    ax.set_ylabel("Relative error at t = t_max")
    ax.set_title("Time-stepping convergence (gaps = diverged; RK45 is adaptive)")
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath)
    return results

def main():
    colormap = "inferno"
    N, L = 64, 2*np.pi
    f = lambda x,y,t: 2.0 * np.sin(x) * np.sin(y) * (np.cos(10.0*t) - 5.0 * np.sin(10.0 * t))
    u0 = lambda x,y,t: np.sin(x) * np.sin(y)
    u_exact = lambda x,y,t : np.cos(10.0 * t) * np.sin(x) * np.sin(y)
    t_max = 2.0*np.pi
    t_steps = 2000

    # Get solutions
    #sol_FE = heatSolver(N, L, f, u0, t_max, t_steps, method="FE")
    #sol_FE = heatSolver(N, L, f, u0, t_max, method="FE")
    #sol_FE = heatSolver(N, L, f, u0, t_max, t_steps, method="BE")
    #sol_FE = heatSolver(N, L, f, u0, t_max, t_steps, method="CN")
    #sol_FE = heatSolver(N, L, f, u0, t_max, method="RK2")
    #sol_FE = heatSolver(N, L, f, u0, t_max, t_steps, method="BDF2")
    sol_FE = heatSolver(N, L, f, u0, t_max, method="RK45")
    sol_exact = u_exact(sol_FE[0], sol_FE[1], t_max)
    err = sol_FE[2][-1] - sol_exact

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(sol_FE[0], sol_FE[1], err, cmap=colormap)
    fig.colorbar(surf, shrink=0.6, aspect=12, label="z")
    plt.show()

    # Convergence study for all methods
    convTest(u_exact, f, u0, t_max, L)

    return 0

if __name__ == "__main__":
    main()