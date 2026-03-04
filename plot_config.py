# Shared color palette for prevalence estimation methods.
# Paul Tol colorblind-friendly palette with alternating cool/warm tones.

METHOD_COLORS = {
    # Names used in simulation notebook
    "Uncalibrated": "#4477AA",        # blue
    "Classify & Count": "#EE6677",    # rose
    "Rogan-Gladen": "#228833",        # green
    "SLD (EMQ)": "#66CCEE",           # cyan
    "Global Calibration": "#AA3377",  # purple
    "Multicalibration": "#222222",    # near-black
    # Additional names used in ACS notebook
    "Raw Scores": "#4477AA",          # blue  (= Uncalibrated)
    "PACC": "#CCBB44",                # yellow
    "Isotonic Regression": "#AA3377", # purple (= Global Calibration)
    "MCGrad": "#222222",              # near-black (= Multicalibration)
    "MCGrad + Emb.": "#44AA99",       # teal
}
