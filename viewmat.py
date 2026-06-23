from scipy.io import loadmat
import pandas as pd
import numpy as np

mat = loadmat("./NASA /4. BatteryAgingARC_45_46_47_48/B0048.mat")

cycles = mat["B0048"]["cycle"][0, 0]

all_rows = []

for cycle_idx in range(cycles.shape[1]):

    cycle = cycles[0, cycle_idx]

    cycle_type = str(cycle["type"][0])

    ambient_temp = float(cycle["ambient_temperature"][0][0])

    data = cycle["data"][0, 0]

    fields = data.dtype.names

    # Find first non-empty field
    n = None
    for field in fields:
        try:
            val = data[field]
            if val.size > 0:
                arr = np.array(val).squeeze()
                if arr.size > 0:
                    n = arr.size
                    break
        except Exception:
            pass

    if n is None:
        continue

    for j in range(n):

        row = {
            "cycle": cycle_idx + 1,
            "type": cycle_type,
            "ambient_temperature": ambient_temp,
        }

        for field in fields:

            try:
                val = data[field]

                if val.size == 0:
                    row[field] = None
                    continue

                arr = np.array(val).squeeze().flatten()

                row[field] = arr[j] if j < len(arr) else None

            except Exception:
                row[field] = None

        all_rows.append(row)

df = pd.DataFrame(all_rows)

df.to_csv("B0048_full.csv", index=False)

print("Saved", len(df), "rows")