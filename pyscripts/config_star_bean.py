# config_star_bean.py
#
# Config central para STAR-like-tokamak-equilibrium
# - Targets geométricos (Miller-like)
# - Parámetros de equilibrio
# - Import CAD + plasma target AUTO
# - Blanket filaments (pasivos)
# - Knobs del objective usado por scan_star_refine.py (CAD/diag)
# - Constraint LCFS dentro de WALL_INNER
# - NUEVO: límites recomendados de corriente por familia (Imax) + límite operativo

# ----------------------------
# Target geometry (Miller-like)
# ----------------------------
R0_geom    = 4.0
A_geom     = 2.0
kappa_geom = 2.23
delta_geom = 0.62

# ----------------------------
# Coil current grouping / distribution
# ----------------------------
coil_group_mode = "area"   # "area" recomendado ("equal" también posible)

# ----------------------------
# PF / CS currents (total family currents) [A]
# ----------------------------
# CS_current  = 6.730713e5
# PF1_current = -2.003520e5
# PF2_current = -6.612579e5
# PF3_current = -0.0e6
#
# # Familias extra para divertor / strike-point control
# PF4_current = 0.0e6
# PF5_current = 0.0e6
# PF6_current = 0.0e6

# --- seed from scan_star_feasibility (f=0.2) ---
# CS_current  = -20.899566993235915e6
# PF1_current = -10.512417839899967e6
# PF2_current =  2.072127369478877e6
# PF3_current =  4.727175728636371e6
# PF4_current =  2.5975154813371883e6
# PF5_current =  0.8970283883312203e6
# PF6_current = -2.12163640024167e6

# PF2_current = -3.672210310794951e6
# PF3_current =  2.185249470608207e6
# PF4_current = -0.0007183742469442367e6
# PF5_current = -1.852723815778954e6
# PF6_current =  5.062727735707545e6
# CS_current  =  1.9999816859808093e6
# PF1_current = -1.5519747091956937e6

# CS_current = -12e6
# PF1_current = -1.5519747092e6
# PF2_current = -3.27221031079e6
# PF3_current = 1.88524947061e6
# PF4_current = 1.19928162575e6
# PF5_current = -1.45272381578e6
# PF6_current = 8.56272773571e6

# CS_current = -25e6
# PF1_current = -1.5519747092e6
# PF2_current = -3.27221031079e6
# PF3_current = 1.88524947061e6
# PF4_current = 0.6992816257499999e6
# PF5_current = -1.45272381578e6
# PF6_current = 8.56272773571e6

# CS_current = -40e6
# PF1_current = -1.5519747092e6
# PF2_current = -3.27221031079e6
# PF3_current = 1.88524947061e6
# PF4_current = 1.9492816257499999e6
# PF5_current = -1.32772381578e6
# PF6_current = 8.31272773571e6

# CS_current = -40e6
# PF1_current = -1.5519747092e6
# PF2_current = -3.27221031079e6
# PF3_current = 1.88524947061e6
# PF4_current = 2.94928162575e6
# PF5_current = -1.26522381578e6
# PF6_current = 8.34397773571e6

# CS_current  =  1.9999816859808093e6
# PF1_current = -1.5519747091956937e6
# PF2_current = -3.672210310794951e6
# PF3_current =  2.185249470608207e6
# PF4_current = -0.0007183742469442367e6
# PF5_current = -1.852723815778954e6
# PF6_current =  5.062727735707545e6

# CS_current  =  -4500000.0*(5)
# PF1_current = -1551471.3269405721
# PF2_current = -3240319.0777849997
# PF3_current =  1677766.7143307133
# PF4_current = 6659703.18739089*(2)
# PF5_current =  4864393.41851279*(-0.8)
# PF6_current =  12124522.848095525*(0.3)

# CS_current  =  -4500000.0
# PF1_current = -1551471.3269405721
# PF2_current = -3240319.0777849997
# PF3_current =  1677766.7143307133
# PF4_current = -6659703.18739089
# PF5_current =  4864393.4185127
# PF6_current =  12124522.848095525

# CS_current  =  -5500000.0*(1.0)
# PF1_current = -1755849.8983813403
# PF2_current =  -2998086.6201337054*(1.1)
# PF3_current =  1692680.4443442654
# PF4_current = 6071663.348636366
# PF5_current =  5376091.303354473
# PF6_current =  11263833.721367357*(0.87)

# CS_current  =  -5500000.0*(150.0)
# PF1_current = -1755849.8983813403
# PF2_current =  -2998086.6201337054*(1.1)
# PF3_current =  1692680.4443442654*(1.0)
# PF4_current = 6071663.348636366*(1.2)
# PF5_current =  5376091.303354473
# PF6_current =  11263833.721367357*(0.17)

# CS_MID_current  = -3.575e6
# CS_END_current  = -1.925e6
# CS_current  =  -5500000.0*(0.0)
# PF1_current = -1755849.8983813403
# PF2_current =  -2998086.6201337054*(1.1)
# PF3_current =  1692680.4443442654
# PF4_current = 6071663.348636366
# PF5_current =  5376091.303354473
# PF6_current =  11263833.721367357*(0.87)

# CS_MID_current  = -4627491.5266349185*(10.0)
# CS_END_current  = -1465360.7175870144*(1.0)
# CS_current  =  -5500000.0*(0.0)
# PF1_current = -1783755.5861325455
# PF2_current =  -2932924.101598065*(1.1)
# PF3_current =  1725784.107504085
# PF4_current = 6041143.335032868
# PF5_current =  5575173.286187235*(3.0)
# PF6_current =  10862894.864683202*(1.3)

# CS_MID_current  = -4627491.5266349185*(10.0)
# CS_END_current  = -1465360.7175870144*(1.0)
# CS_current  =  -5500000.0*(0.0)
# PF1_current = -1783755.5861325455
# PF2_current =  -2932924.101598065*(1.1)
# PF3_current =  1725784.107504085
# PF4_current = 6041143.335032868*(1.0)
# PF5_current =  5575173.286187235*(3.0)
# PF6_current =  10862894.864683202*(1.2)

# CS_MID_current  = -4627491.5266349185*(7.0) # 5.0
# CS_END_current  = -1465360.7175870144*(-15.0)
# CS_current  =  -5500000.0*(0.0)
# PF1_current = -1783755.5861325455
# PF2_current =  -2932924.101598065*(1.1)
# PF3_current =  1725784.107504085*(1.0)
# PF4_current = 6041143.335032868*(1.0)
# PF5_current =  5575173.286187235*(6.0)
# PF6_current =  10862894.864683202*(1.2)

# CS_MID_current  = -4627491.5266349185*(5.0)
# CS_END_current  = -1465360.7175870144*(-20.0)
# CS_current  =  -5500000.0*(0.0)
# PF1_current = -1783755.5861325455
# PF2_current =  -2932924.101598065*(1.1)
# PF3_current =  1725784.107504085*(1.0)
# PF4_current = 6041143.335032868*(1.0)
# PF5_current =  5575173.286187235*(5.0)
# PF6_current =  10862894.864683202*(1.2)

CS_MID_current  = -4627491.5266349185*(7.0)
CS_END_current  = -1465360.7175870144*(-20.0)
CS_current  =  -5500000.0*(0.0)
PF1_current = -1783755.5861325455
PF2_current =  -2932924.101598065*(1.11)
PF3_current =  1725784.107504085*(1.0)
PF4_current = 6041143.335032868*(1.0)
PF5_current =  5575173.286187235*(5.0)
PF6_current =  10862894.864683202*(1.2)



BASE_FAMILY_CURRENTS_A = {
    "CS":  CS_current,
    "CS_MID": CS_MID_current,
    "CS_END": CS_END_current,
    "PF1": PF1_current,
    "PF2": PF2_current,
    "PF3": PF3_current,
    "PF4": PF4_current,
    "PF5": PF5_current,
    "PF6": PF6_current,
}

# ------------------------------------------------------------
# NUEVO: max recommended currents (from CAD area + Jeng*Aeff)
# ------------------------------------------------------------
# Estos valores vienen de tu output:
# fill_factor=0.75, Jeng=40 A/mm^2, family_mode=min
MAX_RECOMMENDED_FAMILY_CURRENTS_A = {
    "CS":  73.036e6,
    "PF1": 15.000e6,
    "PF2": 15.000e6,
    "PF3":  7.500e6,
    "PF4":  9.437e6,
    "PF5": 11.850e6,
    "PF6": 19.752e6,
}

Imax_recommended_MA = {
    "CS":  73.036,
    "PF1": 15.000,
    "PF2": 15.000,
    "PF3": 7.500,
    "PF4": 9.437,
    "PF5": 11.850,
    "PF6": 19.752,
}

# Límite operativo recomendado (safety factor conservador)
# - 0.25–0.40 típico si tu Imax viene de supuestos “engineering” y aún no modelas
#   límites térmicos/estructurales del diseño real.
OPERATING_I_SAFETY_FACTOR = 0.35
OPERATING_FAMILY_CURRENT_LIMIT_A = {
    k: OPERATING_I_SAFETY_FACTOR * v for k, v in MAX_RECOMMENDED_FAMILY_CURRENTS_A.items()
}

# Si quieres hard-clip durante scans (si tu scan/refine lo implementa), usa esto:
ENFORCE_OPERATING_LIMITS = False  # True => clamp/castigo en objective

# ----------------------------
# Plasma and profile parameters
# ----------------------------
Ip      = 4.0e6 # 1.72e6
paxis   = 2.0e3 # Pa 1e5
fvac    = 20.8
alpha_m = 1.5
alpha_n = 1.1

vacuum_only = False   # True => Ip=0 y paxis=0

# ----------------------------
# Numerical grid and domain
# ----------------------------
nx_eq     = 65 # 129 65
ny_eq     = 129 # 257 129
margin_RZ = 0.5

# ----------------------------
# Newton–Krylov solver
# ----------------------------
target_rel_tol = 5e-5
target_rel_tol_ramp = 2e-5
f_list_equilibrium = (0.20, 0.40, 0.70, 1.0)

# ----------------------------
# CAD import (MEJORADO DE VERDAD)
# ----------------------------
unit_scale = None

# Resample siempre (si quieres suavidad visual/geométrica real)
resample_walls = "always"

# Más puntos => inner wall más suave + objetivos shape más estables
n_wall   = 1601*0.5
n_inner  = 2001*0.5
n_plasma = 501

min_wall_pts = 100  # ya no manda tanto si resample="always"
enforce_ccw = True
canonical_start = True

# CLAVE: aplanado de bulges/arcs/splines
prefer_path_flattening = True
flatten_distance = 0.002  # bajar esto mejora mucho si el DXF usa bulges/arcos

# CLAVE: densificado por longitud de segmento (arregla low-poly aunque no haya bulges)
max_seg_len_wall = 0.008
max_seg_len_plasma = 0.008

label_match_factor = 2.0

# Plasma target AUTO
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
blanket_enabled = False
blanket_n_filaments = 2500
blanket_distribution = "stratified"
blanket_seed = 0
blanket_wall_margin_m = 0.01
blanket_filament_dR = 0.004
blanket_filament_dZ = 0.004
blanket_bins_R = 0
blanket_bins_Z = 0
blanket_pitch_mode = "auto"
blanket_pitch_R = 0.03
blanket_pitch_Z = 0.03
blanket_label_prefix = "BLK"
blanket_containment_radius = -1e-9

# ----------------------------
# Coil Imax estimation knobs (if you want to recompute)
# ----------------------------
coil_fill_factor = 0.75
coil_Jeng_A_per_mm2 = 40.0
coil_Jeng_by_family = None
coil_family_Imax_mode = "min"

# ----------------------------
# Objective knobs (scan_star_refine.py)
# ----------------------------
sig_R0_m  = 0.25
sig_A     = 0.25
sig_kappa = 0.25
sig_delta = 0.20

sig_x_m     = 0.20
sig_shape_m = 0.05

w_scalar = 1.0
w_x      = 1.0
w_shape  = 1.0

penalty_no_separatrix  = 1e6
penalty_no_xpoints     = 1e6
penalty_fallback_lcfs  = 5e4
penalty_neg_delta      = 10.0

improve_eps = 1e-9

# ----------------------------
# Constraint LCFS dentro de WALL_INNER
# ----------------------------
enforce_inner_wall = True
inner_wall_frac_tol = 0.01
inner_wall_hard_fail = False
penalty_outside_inner = 5e4
penalty_outside_inner_hard = 1e9
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
