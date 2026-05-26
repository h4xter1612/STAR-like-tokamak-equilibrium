# Best current snippet from adaptive_polish_star_highp.py
cs_segmented = True
cs_mid_fraction = 0.45
cs_segment_zcut_m = None
cs_segment_keep_parent = True

CS_current = 0
CS_MID_current = -32519721.3168
CS_END_current = -21859555.4395
PF1_current = -7845651.36141
PF2_current = -7717383.29573
PF3_current = 2340649.27325
PF4_current = 3138100.68358
PF5_current = 6495390.16869
PF6_current = 8331115.85119

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
