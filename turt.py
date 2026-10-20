#!/usr/bin/env python3
"""
turt.py -- launcher for TURT (Tornadic-vortex Utility Radar Toolkit).

    python turt.py            # synthetic demo volume
    python turt.py my_file.nc # real radar file (Py-ART reads it)

(Equivalent: python -m turt [file])

Requires: arm_pyart, numpy, matplotlib; scipy is optional (needed for the Rankine fit).
"""
from turt.launch import launch

if __name__ == "__main__":
    launch()
