from pathlib import Path
import pandas as pd

#ot= 15
#hw =[1, 2, 4, 6, 12]
#
#for hw in hw:
#    data_dir = Path(f"TERZ_HHZ_ont{ot}/TERZ_spectrograms_matrix_results_cut_hw{hw}_ont{ot}/parquet_parts/")
#    df_total = pd.concat(pd.read_parquet(parquet_file) for parquet_file in data_dir.glob('*.parquet'))
#    df_2022 = df_total[df_total['year'] == "2022"]
#    df_2023 = df_total[df_total['year'] == "2023"]
#    df_2024 = df_total[df_total['year'] == "2024"]
#    df_2025 = df_total[df_total['year'] == "2025"]
#    df_2022.to_parquet(f"year_TERZ_parquets/TERZ_HHZ_hw{hw}_ont{ot}_2022.parquet")
#    df_2023.to_parquet(f"year_TERZ_parquets/TERZ_HHZ_hw{hw}_ont{ot}_2023.parquet")
#    df_2024.to_parquet(f"year_TERZ_parquets/TERZ_HHZ_hw{hw}_ont{ot}_2024.parquet")
#    df_2025.to_parquet(f"year_TERZ_parquets/TERZ_HHZ_hw{hw}_ont{ot}_2025.parquet")
#    df_2022.head(5)
#    df_2023.head(5)
#    df_2024.head(5)
#    df_2025.head(5)
#
#hw =[1, 2, 4, 6, 12]
#for hw in hw:
#    data_dir = Path(f"TERZ_HHZ_ont{ot}_resume/TERZ_spectrograms_matrix_results_cut_hw{hw}_ont{ot}_resume/parquet_parts/")
#    df_total = pd.concat(pd.read_parquet(parquet_file) for parquet_file in data_dir.glob('*.parquet'))
#    df_2024 = df_total[df_total['year'] == "2024"]
#    df_2025 = df_total[df_total['year'] == "2025"]
#    df_2024.to_parquet(f"year_TERZ_parquets/TERZ_HHZ_hw{hw}_ont{ot}_resume_2024.parquet")
#    df_2025.to_parquet(f"year_TERZ_parquets/TERZ_HHZ_hw{hw}_ont{ot}_resume_2025.parquet")
#    df_2024.head(5)
#    df_2025.head(5)
#
#hw =[1, 2, 4, 6, 12]
#ot= 20
#for hw in hw:
#    data_dir = Path(f"TERZ_HHZ_ont{ot}/TERZ_spectrograms_matrix_results_cut_hw{hw}_ont{ot}/parquet_parts/")
#    df_total = pd.concat(pd.read_parquet(parquet_file) for parquet_file in data_dir.glob('*.parquet'))
#    df_2022 = df_total[df_total['year'] == "2022"]
#    df_2023 = df_total[df_total['year'] == "2023"]
#    df_2024 = df_total[df_total['year'] == "2024"]
#    df_2025 = df_total[df_total['year'] == "2025"]
#    df_2022.to_parquet(f"year_TERZ_parquets/TERZ_HHZ_hw{hw}_ont{ot}_2022.parquet")
#    df_2023.to_parquet(f"year_TERZ_parquets/TERZ_HHZ_hw{hw}_ont{ot}_2023.parquet")
#    df_2024.to_parquet(f"year_TERZ_parquets/TERZ_HHZ_hw{hw}_ont{ot}_2024.parquet")
#    df_2025.to_parquet(f"year_TERZ_parquets/TERZ_HHZ_hw{hw}_ont{ot}_2025.parquet")
#    df_2022.head(5)
#    df_2023.head(5)
#    df_2024.head(5)
#    df_2025.head(5)

hw =[1, 2, 4, 6, 12]
ot= 25
for hw in hw:
    data_dir = Path(f"TERZ_HHE_ont{ot}/TERZ_spectrograms_matrix_results_cut_hw{hw}_ont{ot}/parquet_parts/")
    df_total = pd.concat(pd.read_parquet(parquet_file) for parquet_file in data_dir.glob('*.parquet'))
    df_2022 = df_total[df_total['year'] == "2022"]
    df_2023 = df_total[df_total['year'] == "2023"]
    df_2024 = df_total[df_total['year'] == "2024"]
    df_2025 = df_total[df_total['year'] == "2025"]
    df_2022.to_parquet(f"year_TERZ_parquets/TERZ_HHE_hw{hw}_ont{ot}_2022.parquet")
    df_2023.to_parquet(f"year_TERZ_parquets/TERZ_HHE_hw{hw}_ont{ot}_2023.parquet")
    df_2024.to_parquet(f"year_TERZ_parquets/TERZ_HHE_hw{hw}_ont{ot}_2024.parquet")
    df_2025.to_parquet(f"year_TERZ_parquets/TERZ_HHE_hw{hw}_ont{ot}_2025.parquet")
    df_2022.head(5)
    df_2023.head(5)
    df_2024.head(5)
    df_2025.head(5)

