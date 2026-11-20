import pandas as pd

df = pd.read_parquet("year_MNSENA_results_cut/qtransform_features.parquet")

df_2022 = df[df['year'] == "2022"]
df_2023 = df[df['year'] == "2023"]
df_2024 = df[df['year'] == "2024"]
df_2025 = df[df['year'] == "2025"]

df_2022.to_parquet("year_MNSENA_results_cut/qtransform_features_2022.parquet")
df_2023.to_parquet("year_MNSENA_results_cut/qtransform_features_2023.parquet")
df_2024.to_parquet("year_MNSENA_results_cut/qtransform_features_2024.parquet")
df_2025.to_parquet("year_MNSENA_results_cut/qtransform_features_2025.parquet")

df_2022.head(5)
df_2023.head(5)
df_2024.head(5)
df_2025.head(5)