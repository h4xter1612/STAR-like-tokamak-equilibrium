# config_star_bean.py
"""
Reference parameters for the refined STAR-like bean equilibrium.

All other scripts (equilibrium solver, diagnostics, scans, animations)
should import from this module so that the scenario is defined in a single
place.
"""

# ----------------------------
# Target geometry (Miller-like)
# ----------------------------
R0_geom    = 4.0      # [m] target major radius
A_geom     = 1.7      # target aspect ratio
kappa_geom = 1.8      # target elongation
delta_geom = 0.30     # target triangularity

# ----------------------------
# PF / CS currents
# (refined STAR-like bean equilibrium)
# ----------------------------
CS_current  = 1.366415e6   # [A]  0.800 MA
PF1_current = -6.792904e5 # [A] -0.230 MA (PF1U/L)
PF2_current = -2.258759e5      # [A]  0.000 MA (PF2U/L off)
PF3_current = 7.360064e5   # [A]  1.100 MA (PF3U/L)

# ----------------------------
# Plasma and profile parameters
# ----------------------------
Ip      = 8.0e5   # [A] plasma current (0.8 MA)
paxis   = 2.0e3   # [Pa] on-axis pressure (2 kPa)
fvac    = 0.5     # vacuum / toroidal field parameter
alpha_m = 1.8     # pressure profile exponent
alpha_n = 1.2     # f(psi) profile exponent

# ----------------------------
# Numerical grid and domain
# ----------------------------
nx_eq     = 65   # number of points in R
ny_eq     = 129   # number of points in Z
margin_RZ = 0.5   # [m] extra margin around the outer wall

# ----------------------------
# Newton–Krylov solver
# ----------------------------
target_rel_tol = 1e-5  # target relative tolerance for the GS solve

# ----------------------------
# Output figure / file names
# ----------------------------
fig_equilibrium = "STAR_bean_equilibrium.png"
fig_shape       = "STAR_bean_shape.png"
fig_q_profile   = "STAR_bean_q_profile.png"
fig_pressure    = "STAR_bean_pressure.png"
fig_jtor        = "STAR_bean_jtor_map.png"
fig_shear       = "STAR_bean_shear_profile.png"
txt_global      = "STAR_bean_global_numbers.txt"
