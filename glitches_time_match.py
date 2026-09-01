import pandas as pd
from obspy import UTCDateTime

GPS_UNIX_OFFSET = 315964800
GPS_UTC_OFFSET = 18
WINDOW_SECONDS = 12

# -------------------------------------------------------------------
# Read data
# -------------------------------------------------------------------

df_s = pd.read_parquet("Virgo_results_cut/O3bqtransform_features.parquet")
df_g = pd.read_csv("gravity_spy_O3b.csv")

# Convert GPS time -> UTC
df_g["trigger_time"] = df_g["GPStime"].apply(lambda gps: UTCDateTime(gps + GPS_UNIX_OFFSET - GPS_UTC_OFFSET).datetime)

# Keep only required columns
df_s = df_s[["trigger_time", "file"]].copy()
df_g = df_g[["trigger_time", "label"]].copy()

# -------------------------------------------------------------------
# Convert both trigger columns to timezone-aware pandas datetimes
# -------------------------------------------------------------------

df_s["trigger_time"] = pd.to_datetime(df_s["trigger_time"], utc=True)
df_g["trigger_time"] = pd.to_datetime(df_g["trigger_time"], utc=True)

# Sort
df_s = df_s.sort_values("trigger_time").reset_index(drop=True)
df_g = df_g.sort_values("trigger_time").reset_index(drop=True)

#print(df_s["trigger_time"].head(5))
#print()
#print(df_g["trigger_time"].head(5))
#print()

g_times = df_g["trigger_time"].to_numpy()
s_times = df_s["trigger_time"].to_numpy()

window = pd.Timedelta(seconds=WINDOW_SECONDS).to_numpy()

matched_rows = []

matched_s_indices = set()
matched_g_indices = set()

for s_idx, s_time in enumerate(s_times):
    left = s_time - window
    right = s_time + window

    start = g_times.searchsorted(left, side="left") # Indices of all Gravity Spy triggers >= left
    end = g_times.searchsorted(right, side="right") # Indices of all Gravity Spy triggers <= right
    g_indices = range(start, end) # All Gravity Spy triggers inside [s_time-12s, s_time+12s]

    for g_idx in g_indices:

        matched_rows.append({"file": df_s.iloc[s_idx]["file"],"trigger (df_s)": df_s.iloc[s_idx]["trigger_time"],
                             "trigger (df_g)": df_g.iloc[g_idx]["trigger_time"],"label": df_g.iloc[g_idx]["label"],})

        matched_s_indices.add(s_idx)
        matched_g_indices.add(g_idx)


df_matched = pd.DataFrame(matched_rows,columns=["file","trigger (df_s)","trigger (df_g)","label"])

df_g_unmatched = df_g.loc[~df_g.index.isin(matched_g_indices)].copy()
df_s_unmatched = df_s.loc[~df_s.index.isin(matched_s_indices)].copy()

df_matched = df_matched.reset_index(drop=True)
df_g_unmatched = df_g_unmatched.reset_index(drop=True)
df_s_unmatched = df_s_unmatched.reset_index(drop=True)


print(f"Matched pairs:       {len(df_matched)}")
print(f"Unmatched df_g:      {len(df_g_unmatched)}")
print(f"Unmatched df_s:      {len(df_s_unmatched)}")
