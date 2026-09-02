from pathlib import Path
import pandas as pd
from obspy import UTCDateTime
import matplotlib.pyplot as plt

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
df_matched = df_matched.reset_index(drop=True)
#print(df_matched.head(5))
#print(df_matched['label'].unique())

base_dir = Path("event_histograms_output")
output_dir = base_dir
counter = 1
while output_dir.exists():
    output_dir = Path(f"{base_dir}_{counter}")
    counter += 1

output_dir.mkdir(parents=True)
print(f"\nOutput directory: {output_dir}")

glitches_order = {"Low_Frequency_Burst": 1,"Scattered_Light": 2,"None_of_the_Above": 3,"Extremely_Loud": 4,
                  "Low_Frequency_Lines": 5,"Paired_Doves": 6,"No_Glitch": 7,"Light_Modulation": 8,
                  "Power_Line": 9,"Scratchy": 10,"Helix": 11,"Whistle": 12,
                  "Wandering_Line": 13,"Violin_Mode": 14,"Air_Compressor": 15,"Tomte": 16,
                  "Koi_Fish": 17,"Blip": 18,"1400Ripples": 19,"Repeating_Blips": 20}


# -------------------------------------------------------------------
# Generate one histogram for each df_s trigger
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Generate one histogram for each df_s trigger
# -------------------------------------------------------------------

# Get all 20 labels ordered by their values in glitches_order
all_ordered_labels = sorted(glitches_order.keys(), key=lambda k: glitches_order[k])

for trigger_s, group in df_matched.groupby("trigger (df_s)", sort=True):

    # Count values and reindex against all 20 labels, filling missing ones with 0
    label_counts = group["label"].value_counts().reindex(all_ordered_labels, fill_value=0)

    file_name = group["file"].iloc[0]
    trigger_string = trigger_s.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(label_counts.index, label_counts.values)
    ax.set_xlabel("Label")
    ax.set_ylabel("Number of incidences")
    ax.set_title(f"{file_name}\n df_s trigger: {trigger_string} UTC")
    
    # Rotate x-axis labels so all 20 label names are legible
    ax.tick_params(axis="x", rotation=45)
    plt.xticks(ha="right")
    
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    trigger_filename = trigger_s.strftime("%Y%m%dT%H%M%S_%f")[:-3]
    output_file = output_dir / f"histogram_{trigger_filename}.pdf"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


print("\nAll histograms generated successfully.")

