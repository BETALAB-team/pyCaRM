# -*- coding: utf-8 -*-
"""
Example: building simulation inputs from an Excel file.

pyCaRM has no built-in Excel reader: ``EnvironmentalTimeSeries`` and
``FieldInput`` are only built from arrays (``from_array``/``from_matrix``).
Reading data from an Excel file is a couple of lines of pandas, so anyone
who wants to keep supplying input via Excel can still do so without the
library depending on pandas/openpyxl. This script writes two small demo
Excel files (standing in for real input data) and then shows the recipe
for reading each one back into pyCaRM.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from carm import EnvironmentalTimeSeries, FieldInput


def main():

    tmp_dir = Path(tempfile.mkdtemp())

    # -------------------------------------------------------------------------
    # Stand-in for real input data: a climate time series and a field layout.
    # -------------------------------------------------------------------------

    n_steps = 24
    angle = np.linspace(0, 2 * np.pi, n_steps)
    env_df = pd.DataFrame(
        {
            "T_ext": 12.0 + 5.0 * np.sin(angle),
            "SolarRad": np.clip(600.0 * np.sin(angle), 0, None),
        }
    )
    env_path = tmp_dir / "input_env.xlsx"
    env_df.to_excel(env_path, index=False)

    field_df = pd.DataFrame({"x": [2.5, 7.5, 2.5, 7.5], "y": [2.5, 2.5, 7.5, 7.5]})
    field_path = tmp_dir / "spacing.xlsx"
    field_df.to_excel(field_path, index=False)

    # -------------------------------------------------------------------------
    # The actual recipe: pandas.read_excel + from_array.
    # -------------------------------------------------------------------------

    env_df = pd.read_excel(env_path)
    env_input = EnvironmentalTimeSeries.from_array(
        Tm=13.0,
        T_ext=env_df["T_ext"].to_numpy(),
        SolarRad=env_df["SolarRad"].to_numpy(),
    )

    field_df = pd.read_excel(field_path)
    myfield = FieldInput(n_bhes=4, xmin=0, ymin=0, xmax=10, ymax=10, rb=0.075)
    myfield.from_array(field_df["x"].to_numpy(), field_df["y"].to_numpy())

    print(f"env_input.T_ext.shape = {env_input.T_ext.shape}")
    print(f"myfield.borehole_coordinates = {myfield.borehole_coordinates}")


if __name__ == "__main__":
    main()
