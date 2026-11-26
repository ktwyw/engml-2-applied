"""Physical constants (CODATA 2018) and curated pure-component molecular reference dataset."""

R = 8.314462618             # J/(mol K)
k_B = 1.380649e-23          # J/K
N_A = 6.02214076e23         # 1/mol
T_STD = 298.15              # K
P_STD = 1.0e5               # Pa

# Reference homologous alkanes & polar molecules: (n_C, n_H, n_O, Mw [g/mol], Tb [K], Tc [K], Pc [MPa], omega [-])
MOLECULES_DB = {
    "methane":   {"nC": 1,  "nH": 4,  "nO": 0, "Mw": 16.043,  "Tb": 111.66, "Tc": 190.56, "Pc": 4.599, "omega": 0.0115},
    "ethane":    {"nC": 2,  "nH": 6,  "nO": 0, "Mw": 30.070,  "Tb": 184.55, "Tc": 305.32, "Pc": 4.872, "omega": 0.0995},
    "propane":   {"nC": 3,  "nH": 8,  "nO": 0, "Mw": 44.097,  "Tb": 231.04, "Tc": 369.83, "Pc": 4.248, "omega": 0.1523},
    "n-butane":  {"nC": 4,  "nH": 10, "nO": 0, "Mw": 58.123,  "Tb": 272.66, "Tc": 425.12, "Pc": 3.796, "omega": 0.2002},
    "n-pentane": {"nC": 5,  "nH": 12, "nO": 0, "Mw": 72.150,  "Tb": 309.22, "Tc": 469.70, "Pc": 3.370, "omega": 0.2515},
    "n-hexane":  {"nC": 6,  "nH": 14, "nO": 0, "Mw": 86.177,  "Tb": 341.88, "Tc": 507.60, "Pc": 3.025, "omega": 0.3013},
    "n-heptane": {"nC": 7,  "nH": 16, "nO": 0, "Mw": 100.204, "Tb": 371.57, "Tc": 540.20, "Pc": 2.740, "omega": 0.3495},
    "n-octane":  {"nC": 8,  "nH": 18, "nO": 0, "Mw": 114.231, "Tb": 398.82, "Tc": 568.70, "Pc": 2.490, "omega": 0.3996},
    "n-nonane":  {"nC": 9,  "nH": 20, "nO": 0, "Mw": 128.258, "Tb": 423.97, "Tc": 594.60, "Pc": 2.290, "omega": 0.4435},
    "n-decane":  {"nC": 10, "nH": 22, "nO": 0, "Mw": 142.285, "Tb": 447.30, "Tc": 617.70, "Pc": 2.110, "omega": 0.4923},
    "isobutane": {"nC": 4,  "nH": 10, "nO": 0, "Mw": 58.123,  "Tb": 261.40, "Tc": 407.85, "Pc": 3.629, "omega": 0.1840},
    "neopentane": {"nC": 5, "nH": 12, "nO": 0, "Mw": 72.150,  "Tb": 282.65, "Tc": 433.74, "Pc": 3.196, "omega": 0.1961},
    "methanol":  {"nC": 1,  "nH": 4,  "nO": 1, "Mw": 32.042,  "Tb": 337.85, "Tc": 512.64, "Pc": 8.097, "omega": 0.5640},
    "ethanol":   {"nC": 2,  "nH": 6,  "nO": 1, "Mw": 46.069,  "Tb": 351.44, "Tc": 513.92, "Pc": 6.148, "omega": 0.6444},
    "1-propanol": {"nC": 3, "nH": 8,  "nO": 1, "Mw": 60.096,  "Tb": 370.35, "Tc": 536.78, "Pc": 5.175, "omega": 0.6209},
    "1-butanol": {"nC": 4,  "nH": 10, "nO": 1, "Mw": 74.123,  "Tb": 390.88, "Tc": 563.05, "Pc": 4.423, "omega": 0.5902},
}
