import numpy as np


def build_ocv_table_from_cc(cc_data, points=300):
    table_data = cc_data.copy()
    table_data["soc"] = 1.0 - table_data["soc_dod_percent"] / 100.0
    table_data = table_data.sort_values("soc")

    soc = table_data["soc"].to_numpy()
    voltage = table_data["voltage"].to_numpy()

    unique_soc, unique_index = np.unique(soc, return_index=True)
    unique_voltage = voltage[unique_index]

    soc_table = np.linspace(unique_soc.min(), unique_soc.max(), points)
    ocv_table = np.interp(soc_table, unique_soc, unique_voltage)

    return soc_table, ocv_table
