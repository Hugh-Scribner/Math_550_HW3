import matplotlib.pyplot as plt
import numpy as np
import io
import contextlib
from matplotlib.lines import Line2D
from functions import heatSolver

# A lot of the values this problem wants us to find can be pretty nearly read off of the convergence plot (at least to an 
# error of 5%) so I am going to use those points. When the value for the error does not directly pass over a grid point, I did
# Some testing to see if I can find a more accurate value. Values are in the table below:

# 5e-2 - every method made it within constraints using 64x64
# FE: 2508 - diverges for fewer steps (CFL said I needed 2527 to converge - HA!)
# BE: 160
# CN: 81
# Rk2: 2508 - needs validation
# BDF2: 160

# 1e-3 - every method made it within constraints using 64x64
# FE: 6200
# BE: 6200
# CN: 555
# RK2: 2510 - immediatly goes to O(1e-5) with 2 more steps from the 5e-2 test
# BDF2: 1115

# 5e-5 - everythong makes it with 64x64
# FE: Needs more than 300k iterations but got close with 300k, ran 3 mil by mistake, and still only got 3.4e-5
# BE: will need more than 300k iterations per FE test
# CN: 1975 - ask brennan about this one - it looks like we are getting exponential convergence
# RK2: 2800
# BDF2: 3940 - needs validation

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

def main():
    N = 64
    t_steps = 3940
    num_meth = "BDF2"

    L = 2*np.pi
    f = lambda x,y,t: 2.0 * np.sin(x) * np.sin(y) * (np.cos(10.0*t) - 5.0 * np.sin(10.0 * t))
    u0 = lambda x,y,t: np.sin(x) * np.sin(y)
    u_exact = lambda x,y,t : np.cos(10.0 * t) * np.sin(x) * np.sin(y)
    t_max = 2.0*np.pi
    
    
    # Get solutions
    sol = heatSolver(N, L, f, u0, t_max, t_steps, method=num_meth)
    x = sol[0]
    y = sol[1]
    err = rel_error(u_exact(x,y,t_max), sol[2])
    print(f"Error for test is {err}")


if __name__ == "__main__":
    main()