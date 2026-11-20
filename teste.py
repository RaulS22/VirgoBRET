import pandas as pd

# Read and process first file
df_s1 = pd.read_parquet("year_Virgo_results_cut/qtransform_features.parquet")
print(df_s1.shape)
df_s1 = df_s1.drop(columns=["year"])

# Read and process second file
df_s2 = pd.read_parquet("Virgo_results_cut/O3bqtransform_features.parquet")
df_s2 = df_s2[df_s2["channel"] == 'HH3']
df_s2 = df_s2[df_s2["station"] == 'VRG01']
df_s2 = df_s2.drop(columns=["run"])

# Concatenate dataframes together
df_s = pd.concat([df_s1, df_s2], ignore_index=True)

print(df_s.head(5))
print()
print(df_s.shape)  # shape is an attribute, not a method
