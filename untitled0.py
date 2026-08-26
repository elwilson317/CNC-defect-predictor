import pandas as pd

df = pd.read_excel(
    "cncmilingdata2023-2026.xlsx",
    sheet_name="Sensor_Data"
)

print(df.shape)
print(df.head())