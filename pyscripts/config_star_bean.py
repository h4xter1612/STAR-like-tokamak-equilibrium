# config_star_bean.py

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
coil_group_mode = "area"

# ----------------------------
# PF / CS currents (total family currents)
# ----------------------------
CS_current  = 1.996172175407065e6
PF1_current = -1.5519747091956937e6
PF2_current = 1.0844960009693208e6
PF3_current = -0.07933410109009831e6

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
f_list_equilibrium = (0.15, 0.35, 0.65, 1.0)

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
# Output figure / file names
# ----------------------------
fig_equilibrium = "STAR_bean_equilibrium.png"
fig_shape       = "STAR_bean_shape.png"
fig_q_profile   = "STAR_bean_q_profile.png"
fig_pressure    = "STAR_bean_pressure.png"
fig_jtor        = "STAR_bean_jtor_map.png"
fig_shear       = "STAR_bean_shear_profile.png"
txt_global      = "STAR_bean_global_numbers.txt"

