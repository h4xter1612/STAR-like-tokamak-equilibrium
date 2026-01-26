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
# Coil current grouping / distribution
# ----------------------------
# If your CAD has segmented coils (e.g., CS1M, CS2U, CS2L, ...),
# you must NOT set each segment to CS_current, or you'd multiply the CS effect.
# This mode tells the code how to distribute a single family current across segments:
#   - "same"  : each segment gets the full current (NOT recommended for segmented CS)
#   - "equal" : split equally among segments (sum of segment currents = family current)
#   - "area"  : split by segment cross-sectional area (uniform current density proxy)
coil_group_mode = "area"

# ----------------------------
# PF / CS currents (total family currents)
# ----------------------------
CS_current  = 2.000000e5   # [A]  1.485 MA  (TOTAL CS family current)
PF1_current = -3.337859e5  # [A] -0.841 MA  (TOTAL PF1 family current)
PF2_current = -4.944562e5  # [A] -0.219 MA  (TOTAL PF2 family current)
PF3_current = 0.000000e0  # [A]  0.918 MA  (TOTAL PF3 family current)

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
nx_eq     = 65     # number of points in R
ny_eq     = 129    # number of points in Z
margin_RZ = 0.5    # [m] extra margin around the outer wall

# ----------------------------
# Newton–Krylov solver
# ----------------------------
# Final target tolerance (last continuation step)
target_rel_tol = 1e-5

# Slightly looser tolerance for intermediate continuation steps (recommended)
target_rel_tol_ramp = 3e-6

# Continuation schedule for equilibrium solve (robust ramp)
f_list_equilibrium = (0.08, 0.15, 0.25, 0.40, 0.60, 0.78, 0.90, 1.00)

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

