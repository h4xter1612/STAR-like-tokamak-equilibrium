# config_star_bean.py
#
# Config central para STAR-like-tokamak-equilibrium
# - Targets geométricos (Miller-like)
# - Parámetros de equilibrio
# - Import CAD + plasma target AUTO
# - Blanket filaments (pasivos)
# - Knobs del objective usado por scan_star_refine.py (CAD/diag)
# - NUEVO: constraint "LCFS dentro de WALL_INNER"

# ----------------------------
# Target geometry (Miller-like)
# ----------------------------
R0_geom    = 4.0
A_geom     = 2.0
kappa_geom = 2.5
delta_geom = 0.60

# ----------------------------
# Coil current grouping / distribution
# ----------------------------
coil_group_mode = "area"   # "area" recomendado

# ----------------------------
# PF / CS currents (total family currents) [A]
# ----------------------------

CS_current  = 6.730713e5
PF1_current = -2.003520e5
PF2_current = -6.612579e5
PF3_current = 0.000000e0

# ----------------------------
# Plasma and profile parameters
# ----------------------------
Ip      = 8.0e5
paxis   = 2.0e3
fvac    = 0.5
alpha_m = 1.8
alpha_n = 1.2

# Si quieres “equilibrio sin plasma” (vacuum coils-only):
vacuum_only = False   # True => Ip=0 y paxis=0

# ----------------------------
# Numerical grid and domain
# ----------------------------
nx_eq     = 129
ny_eq     = 257
margin_RZ = 0.5

# ----------------------------
# Newton–Krylov solver
# ----------------------------
target_rel_tol = 1e-5
target_rel_tol_ramp = 2e-5

# Continuation (recomendado para robustez):
f_list_equilibrium = (0.15, 0.35, 0.65, 1.0)

# Si quieres SOLO un paso (menos robusto, más fallos):
# OJO: debe ser tupla de 1 elemento -> (1.0,)
# f_list_equilibrium = (1.0,)

# ----------------------------
# CAD import
# ----------------------------
unit_scale = None
resample_walls = "auto"
n_wall = 801
n_inner = 801
n_plasma = 320
min_wall_pts = 200
enforce_ccw = True
canonical_start = True
flatten_distance = 0.01
label_match_factor = 2.0

# Plasma target AUTO (si lo usas)
plasma_target_mode = "auto"
plasma_fit_to_inner_if_available = True
plasma_R0 = 4.0
plasma_A = 2.0
plasma_kappa = 2.5
plasma_Z0 = 0.0
plasma_delta_max = 0.70
plasma_delta_grid = 17
plasma_delta_symmetric = True
plasma_shrink_iters = 20
plasma_scale_safety = 0.999
containment_radius = -1e-9
fix_center_if_outside = True
center_search_samples = 800
center_search_seed = 0
strike_ray_fallback_len = 3.0

# ----------------------------
# BLANKET / passive filaments (AUTO between inner & outer wall)
# ----------------------------
blanket_enabled = True

# número deseado de filamentos
blanket_n_filaments = 2500

# "stratified" (recomendado), "grid", "random"
blanket_distribution = "stratified"

# semilla para reproducibilidad (importante para scans)
blanket_seed = 0

# margen para no pegarse a paredes (m)
blanket_wall_margin_m = 0.01   # 1 cm (ajusta)

# tamaño de cada filamento como rectángulo (half-extents en m)
blanket_filament_dR = 0.004
blanket_filament_dZ = 0.004

# sólo aplica a stratified (si <=0 => auto)
blanket_bins_R = 0
blanket_bins_Z = 0

# sólo aplica a grid
blanket_pitch_mode = "auto"     # "auto" o "manual"
blanket_pitch_R = 0.03          # si manual
blanket_pitch_Z = 0.03

blanket_label_prefix = "BLK"
blanket_containment_radius = -1e-9

# ----------------------------
# Objective knobs (scan_star_refine.py)
# ----------------------------
# Sigmas
sig_R0_m  = 0.25
sig_A     = 0.25
sig_kappa = 0.25
sig_delta = 0.20

sig_x_m     = 0.20
sig_shape_m = 0.05

# Pesos
w_scalar = 1.0
w_x      = 1.0
w_shape  = 1.0

# Penalizaciones base
penalty_no_separatrix  = 1e6
penalty_no_xpoints     = 1e6
penalty_fallback_lcfs  = 5e4
penalty_neg_delta      = 10.0

# Para refine tie-breaking / mejora mínima
improve_eps = 1e-9

# ----------------------------
# NUEVO: constraint LCFS dentro de WALL_INNER
# ----------------------------
# Si no hay WALL_INNER en CAD, esto no aplica (frac_out_inner queda None).
enforce_inner_wall = True

# Tolerancia en fracción de puntos fuera (0.0 = estricto)
inner_wall_frac_tol = 0.0

# Si True -> hard fail si se sale (misfit fijo grande)
inner_wall_hard_fail = False

# Penalización soft: misfit += penalty_outside_inner * frac_out_inner
penalty_outside_inner = 5e5

# Penalización hard (si hard_fail=True)
penalty_outside_inner_hard = 1e9

# Igual que containment_radius: radius<0 hace “estricto” (puntos sobre pared cuentan como fuera)
inner_containment_radius = -1e-9

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

