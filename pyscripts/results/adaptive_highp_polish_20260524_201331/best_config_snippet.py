# Best current snippet from adaptive_polish_star_highp.py
cs_segmented = True
cs_mid_fraction = 0.45
cs_segment_zcut_m = None
cs_segment_keep_parent = True

CS_current = 0
CS_MID_current = -29260571.2295
CS_END_current = -24560339.593
PF1_current = -7790457.88984
PF2_current = -8324445.56241
PF3_current = 2107070.53385
PF4_current = 3022994.05729
PF5_current = 6401534.11317
PF6_current = 8774479.06859

Ip = _env_float("STAR_IP_A", 13200000)
paxis = _env_float("STAR_PAXIS_PA", 1200000)

BASE_FAMILY_CURRENTS_A = {
    "CS": CS_current,
    "CS_MID": CS_MID_current,
    "CS_END": CS_END_current,
    "PF1": PF1_current,
    "PF2": PF2_current,
    "PF3": PF3_current,
    "PF4": PF4_current,
    "PF5": PF5_current,
    "PF6": PF6_current,
}
