import os
import sys
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "breath_check_server.settings")
django.setup()

import numpy as np
import pandas as pd
from diagnostic_app.views import parse_file_content
from preprocessing import SpectraPreprocessor

df = pd.DataFrame({
    'Shift': np.arange(1024),
    'Intensity': np.random.rand(1024)
})
content = df.to_csv(index=False, header=False).encode('utf-8')

runs = parse_file_content(content, "test.csv")
print("Parsed runs:", len(runs))
if runs:
    run = runs[0]
    print("Shift len:", len(run["shift"]))
    print("Intensity len:", len(run["intensity"]))
    
    raw_int = np.array(run["intensity"])
    print("raw_int shape:", raw_int.shape)
    
    ml_b = raw_int - SpectraPreprocessor.baseline_als(raw_int)
    print("ml_b shape:", ml_b.shape)
    
    snv = SpectraPreprocessor.apply_snv([ml_b])
    print("snv shape:", snv.shape)
    
    ml_input = snv[0]
    print("ml_input shape:", ml_input.shape)
    
    print("[ml_input] shape:", np.asarray([ml_input]).shape)
