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
# Environment helpers
# ----------------------------
# Permiten que wrappers/continuation cambien parámetros sin editar este archivo.
# Ejemplos PowerShell:
#   $env:STAR_IP_A="6.5e6"
#   $env:STAR_PAXIS_PA="2.0e3"
#   $env:STAR_PASSIVE_STRUCTURES="1"
#   $env:STAR_EQ_DOMAIN_SOURCE="outer"

import os

def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, None)
    if v is None or str(v).strip() == "":
        return int(default)
    return int(float(v))

def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, None)
    if v is None or str(v).strip() == "":
        return float(default)
    return float(v)

def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name, None)
    if v is None or str(v).strip() == "":
        return str(default)
    return str(v).strip()


# ----------------------------
# Target geometry (Miller-like)
# ----------------------------
R0_geom    = 4.2 # 4.2
A_geom     = 2.0
kappa_geom = 2.20 # 2.2 2.23
delta_geom = 0.46 # 0.46 0.62

# ----------------------------
# Coil current grouping / distribution
# ----------------------------
coil_group_mode = "area"   # "area" recomendado ("equal" también posible)

# ------------------------------------------------------------
# CS segmentation control
# ------------------------------------------------------------
cs_segmented = True          # True: create CS + CS_MID + CS_END groups
cs_mid_fraction = 0.45       # central fraction of full CS height assigned to CS_MID
cs_segment_zcut_m = None     # must be None if using cs_mid_fraction
cs_segment_keep_parent = True

# Deprecated / exploratory segmented-CS controls.
# Keep defined for backward compatibility with old scripts, but do not use.
CS_MID_current  = -3729649.665502105*(8.5)*0 #7.7
CS_END_current  = -2136117.664484214*(7.0)*0
CS_current  =  -5500000.0*(0.0)

# ----------------------------
# PF / CS currents (total family currents) [A]
# ----------------------------

# THESE ARE THE WORKING CURRENTS FOR 2E3 PA AND 4E6 MA
# CS_current  =  -3729649.665502105*(8.5)
PF1_current = -1848443.9026171772*(0.21)
PF2_current =  -2821918.8734831177*(1.55)
PF3_current =  1688015.6292537155*(0.76)
PF4_current = 5239815.487861866*(0.30)
PF5_current =  5004189.204634282*(0.75) #1.1 0.76
PF6_current =  10973208.70827962*(0.59)*0.3

CS_current  =  0.0

# CS_MID_current  = -36917866.175284825*1.0 # 1.5
# CS_END_current  = -15433085.616072046*1.0 # 3
# PF1_current =  -4666262.329753918
# PF2_current =  -5129061.062135117
# PF3_current =  -647239.9444473897
# PF4_current =  3126285.842783206
# PF5_current =  6589634.828991643
# PF6_current =  8822664.927545238

# CS_MID_current  = -36917866.175284825*1.0 # 1.5
# CS_END_current  = -15433085.616072046*1.0 # 3
# PF1_current =  -4666262.329753918*1.24
# PF2_current =  -5129061.062135117*1.24  # 1.23
# PF3_current =  -647239.9444473897*0.8
# PF4_current =  3126285.842783206*1.0
# PF5_current =  6589634.828991643*1.0
# PF6_current =  8822664.927545238*1.0

# CS_MID_current  = -36917866.175284825*0.9 # 1.5 0.9
# CS_END_current  = -15433085.616072046*1.0 # 3
# PF1_current =  -4666262.329753918*1.326 #1.23 0.9
# PF2_current =  -5129061.062135117*1.326 # entre 1 y 1.5, arriba 1.2, ~1.3, menos 1.35, mas 1.32, menos 1.33, menos 1.327 menos 1.326
# PF3_current =  -647239.9444473897*-1 # 8 lo comprime mas
# PF4_current =  3126285.842783206*1.0
# PF5_current =  6589634.828991643*1.0
# PF6_current =  8822664.927545238*1.5

# CS_MID_current  = -36917866.175284825*0.9 # 1.5 0.9
# CS_END_current  = -15433085.616072046*1.5 # 3
# PF1_current =  -4666262.329753918*1.4 #1.23 0.9
# PF2_current =  -5129061.062135117*1.5 
# PF3_current =  -647239.9444473897*-3.5 # 8 lo comprime mas -3.5
# PF4_current =  3126285.842783206*1.0*0
# PF5_current =  6589634.828991643*1.0*0
# PF6_current =  8822664.927545238*1.5

# PRENDES PF45? O BAJAR PF1/2?

# CS_MID_current  = -36917866.175284825*1.5 # 1.5 0.9
# CS_END_current  = -15433085.616072046*1.5 # 3
# PF1_current =  -4666262.329753918*1.4*0.1 #1.23 0.9 1.4    0.1
# PF2_current =  -5129061.062135117*1.5                 # 2.5 y pf1 0.1   1.5*2.3
# PF3_current =  -647239.9444473897*-6 # 8 lo comprime mas -3.5   -3.5  -5
# PF4_current =  3126285.842783206*1.0
# PF5_current =  6589634.828991643*1.0
# PF6_current =  8822664.927545238*1.5 # 1.5


# CS_MID_current  = -36917866.175284825*1.5 # 1.5 0.9
# CS_END_current  = -15433085.616072046*1.5 # 3
# PF1_current =  -4666262.329753918*1.4*0.7 #1.23 0.9 1.4    0.1 0.5 0.9 y 0.8 0.6
# PF2_current =  -5129061.062135117*1.5*1.0                 # 2.5 y pf1 0.1   1.5*2.3/1.1
# PF3_current =  -647239.9444473897*-0.0 # 8 lo comprime mas -3.5   -3.5  -5 -6 | -1?
# PF4_current =  3126285.842783206*1.0
# PF5_current =  6589634.828991643*1.0
# PF6_current =  8822664.927545238*1.5 # 1.5


# CS_MID_current  = -36917866.175284825*1.5 # 1.5 0.9
# CS_END_current  = -15433085.616072046*1.5 # 3
# PF1_current =  -4666262.329753918*1.4*0.7 #1.23 0.9 1.4    0.1 0.5 0.9 y 0.8 0.6
# PF2_current =  -5129061.062135117*1.5*1.5                 # 2.5 y pf1 0.1   1.5*2.3/1.1
# PF3_current =  -647239.9444473897*-3 # 8 lo comprime mas -3.5   -3.5  -5 -6 | -1?0
# PF4_current =  3126285.842783206*1.0*-1
# PF5_current =  6589634.828991643*1.0*-1
# PF6_current =  8822664.927545238*1.5 # 1.5


# CS_MID_current  = -36917866.175284825*1.5 # 1.5 0.9
# CS_END_current  = -15433085.616072046*3 # 3
# PF1_current =  -4666262.329753918*1.4*0.8 #1.23 0.9 1.4    0.1 0.5 0.9 y 0.8 0.6          0.78
# PF2_current =  -5129061.062135117*1.5*1.3                 # 2.5 y pf1 0.1   1.5*2.3/1.1    1.4
# PF3_current =  -647239.9444473897*-5 # 8 lo comprime mas -3.5   -3.5  -5 -6 | -1?0
# PF4_current =  3126285.842783206*1.0*-1
# PF5_current =  6589634.828991643*1.0*-1
# PF6_current =  8822664.927545238*1.5 # 1.5


# NUEVO INTENTO
# CS_MID_current  = -36917866.175284825*0.9 
# CS_END_current  = -15433085.616072046*1.5 
# PF1_current =  -4666262.329753918*1.7 
# PF2_current =  -5129061.062135117*1.6 
# PF3_current =  -647239.9444473897*-3.5 
# PF4_current =  3126285.842783206*1.0
# PF5_current =  6589634.828991643*1.0
# PF6_current =  8822664.927545238*10.0

# CS_MID_current  = -32087068.798705857 * 1.0     # mas 2 menos 2.3
# CS_END_current  = -23428913.151029546 * 1.0     # mas 2 menos 2.3   2.03 y luego muere
# PF1_current =  -10582912.900492676 
# PF2_current =  -4920193.27412647 
# PF3_current =  2794442.9304531687 
# PF4_current =  3453245.547837753
# PF5_current =  5191268.491038531
# PF6_current =  4095619.4982399372               # increase it

BASE_FAMILY_CURRENTS_A = {
    "CS":  CS_current,
    "PF1": PF1_current,
    "PF2": PF2_current,
    "PF3": PF3_current,
    "PF4": PF4_current,
    "PF5": PF5_current,
    "PF6": PF6_current,

    # Kept only so older code that expects the keys does not crash.
    "CS_MID": CS_MID_current,
    "CS_END": CS_END_current,
}

# ------------------------------------------------------------
# NUEVO: max recommended currents (from CAD area + Jeng*Aeff)
# ------------------------------------------------------------
# Estos valores vienen de tu output:
# fill_factor=0.75, Jeng=40 A/mm^2, family_mode=min
MAX_RECOMMENDED_FAMILY_CURRENTS_A = {
    "CS":  71.789e6,
    "PF1": 15.000e6,
    "PF2": 15.000e6,
    "PF3":  7.500e6,
    "PF4":  5.829e6,
    "PF5": 8.261e6,
    "PF6": 15.302e6,

    # Kept only so older code that expects the keys does not crash.
    "CS_MID": 32.903e6,
    "CS_END": 38.886e6,
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
# Valores base del equilibrio bueno actual.
# Los wrappers pueden sobreescribir Ip/paxis con variables de entorno.
Ip      = _env_float("STAR_IP_A", 4e6)      # 1.72e6, 4.0e6, 13.2e6
paxis   = _env_float("STAR_PAXIS_PA", 2e3)  # Pa; baseline actual = 2.0e3 1.2e6
fvac    = _env_float("STAR_FVAC", 20.8)
alpha_m = _env_float("STAR_ALPHA_M", 1.5)
alpha_n = _env_float("STAR_ALPHA_N", 1.1)

vacuum_only = _env_bool("STAR_VACUUM_ONLY", False)   # True => Ip=0 y paxis=0

# ----------------------------
# Numerical grid and domain
# ----------------------------
nx_eq     = _env_int("STAR_NX_EQ", 65)     # 129, 65
ny_eq     = _env_int("STAR_NY_EQ", 129)    # 257, 129
margin_RZ = _env_float("STAR_MARGIN_RZ", 0.5)

# Dominio de equilibrio.
# "outer" preserva el comportamiento histórico basado en WALL_OUTER.
# "machine" permite incluir el envelope de la máquina/pasivos si star_equilibrium lo soporta.
eq_domain_source = _env_str("STAR_EQ_DOMAIN_SOURCE", "outer")
limiter_source = _env_str("STAR_LIMITER_SOURCE", "inner")
machine_wall_source = _env_str("STAR_MACHINE_WALL_SOURCE", "outer")

# ----------------------------
# Newton–Krylov solver
# ----------------------------
target_rel_tol = _env_float("STAR_TARGET_REL_TOL", 5e-5)       # 5e-5, 1e-9
target_rel_tol_ramp = _env_float("STAR_TARGET_REL_TOL_RAMP", 2e-5)
f_list_equilibrium = (0.20, 0.40, 0.70, 1.0)

# ----------------------------
# CAD import (MEJORADO DE VERDAD)
# ----------------------------
unit_scale = None

# Resample siempre (si quieres suavidad visual/geométrica real)
resample_walls = "always"

# Más puntos => inner wall más suave + objetivos shape más estables
# Defaults preserve the effective integer values of the previous config:
# int(1601*0.5)=800 and int(2001*0.5)=1000.
n_wall   = _env_int("STAR_N_WALL", 800)
n_inner  = _env_int("STAR_N_INNER", 1000)
n_plasma = _env_int("STAR_N_PLASMA", 501)

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
# Passive structures / blanket controls
# ----------------------------
# Nueva semántica CAD:
#   WALL_INNER  -> first wall / limiter
#   WALL_OUTER  -> blanket outer wall / back plate
#   STAR_VESSEL -> estructuras pasivas metálicas
#
# Para dejar TODO ACTIVADO de momento:
#   passive_structures_enabled=True
#   passive_use_star_vessel=True
#   plot_passive_filaments=True
#
# Para optimización rápida:
#   $env:STAR_PASSIVE_STRUCTURES="0"
#   $env:STAR_PASSIVE_USE_STAR_VESSEL="0"
#   $env:STAR_PLOT_PASSIVES="0"

passive_structures_enabled = _env_bool("STAR_PASSIVE_STRUCTURES", True)
passive_use_star_vessel = _env_bool("STAR_PASSIVE_USE_STAR_VESSEL", False) # True
passive_target_dR_m = _env_float("STAR_PASSIVE_TARGET_DR_M", 0.10)
passive_target_dZ_m = _env_float("STAR_PASSIVE_TARGET_DZ_M", 0.10)
star_vessel_material = _env_str("STAR_VESSEL_MATERIAL", "SS316L")
star_vessel_resistivity_ohm_m = _env_float("STAR_VESSEL_RESISTIVITY_OHM_M", 0.75e-6)

first_wall_material = _env_str("STAR_FIRST_WALL_MATERIAL", "EUROFER97")
first_wall_resistivity_ohm_m = _env_float("STAR_FIRST_WALL_RESISTIVITY_OHM_M", 1.0e-6)
blanket_outer_material = _env_str("STAR_BLANKET_OUTER_MATERIAL", "EUROFER97")
blanket_outer_resistivity_ohm_m = _env_float("STAR_BLANKET_OUTER_RESISTIVITY_OHM_M", 1.0e-6)

plot_passive_filaments = _env_bool("STAR_PLOT_PASSIVES", True)

# Legacy blanket fill entre WALL_INNER y WALL_OUTER.
# Debe quedarse apagado: ahora STAR_VESSEL se usa como pasivo real y la región
# WALL_INNER-WALL_OUTER es blanket/solid region, no relleno artificial.
blanket_enabled = _env_bool("STAR_LEGACY_BLANKET_ENABLED", False)
blanket_n_filaments = _env_int("STAR_LEGACY_BLANKET_N", 0)
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
