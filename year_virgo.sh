set -u

BASE_DIR="year_Virgo_data"

# ------------------------------------------------------------
# Function to download one observing period
# ------------------------------------------------------------

download_period() {

    local PERIOD_NAME="$1"
    local START="$2"
    local END="$3"

    local START_DATE="${START:0:10}"
    local END_DATE="${END:0:10}"

    local OUTDIR="${BASE_DIR}/${PERIOD_NAME}"

    mkdir -p "$OUTDIR"

    local current="$START_DATE"

    while [[ "$current" < "$END_DATE" || "$current" == "$END_DATE" ]]; do

        # ----------------------------------------------------
        # Define default beginning/end of this UTC day
        # ----------------------------------------------------

        local day_start="${current}T00:00:00Z"
        local day_end

        day_end=$(date -u -d "$current + 1 day" \
                  +"%Y-%m-%dT00:00:00Z")

        # ----------------------------------------------------
        # First day: use exact requested start time
        # ----------------------------------------------------

        if [[ "$current" == "$START_DATE" ]]; then
            day_start="$START"
        fi

        # ----------------------------------------------------
        # Last day: use exact requested end time
        # ----------------------------------------------------

        if [[ "$current" == "$END_DATE" ]]; then
            day_end="$END"
        fi

        echo
        echo "============================================================"
        echo "Period : $PERIOD_NAME"
        echo "Date   : $current"
        echo "Start  : $day_start"
        echo "End    : $day_end"
        echo "============================================================"

        # ----------------------------------------------------
        # Stations and channels
        # ----------------------------------------------------

        for station in VRG01; do

            for channel in HH3; do

                outfile="${OUTDIR}/${station}_${channel}_${current}.mseed"

                echo
                echo "------------------------------------------------------------"
                echo "Station : $station"
                echo "Channel : $channel"
                echo "Output  : $outfile"
                echo "------------------------------------------------------------"

                # ------------------------------------------------
                # Skip existing non-empty files
                # ------------------------------------------------

                if [[ -s "$outfile" ]]; then
                    echo "File already exists. Skipping."
                    continue
                fi

                # ------------------------------------------------
                # Download
                # ------------------------------------------------

                fdsnws_fetch \
                    -N 'VR' \
                    -S "$station" \
                    -L '--' \
                    -C "$channel" \
                    -s "$day_start" \
                    -e "$day_end" \
                    -v \
                    -o "$outfile"

                # ------------------------------------------------
                # Check whether download succeeded
                # ------------------------------------------------

                if [[ $? -eq 0 ]]; then
                    echo "Download completed successfully."
                else
                    echo "WARNING: download failed."
                    echo "        $station.$channel on $current"
                fi

            done
        done

        # --------------------------------------------------------
        # Advance by one UTC day
        # --------------------------------------------------------

        current=$(date -u -d "$current + 1 day" \
                  +"%Y-%m-%d")

    done
}


# ============================================================
# 2022
# ============================================================

download_period \
    "2022" \
    "2022-01-01T00:00:00Z" \
    "2022-12-31T23:59:59Z"


# ============================================================
# 2023
# ============================================================

download_period \
    "2023" \
    "2023-01-01T00:00:00Z" \
    "2023-12-31T23:59:59Z"


# ============================================================
# 2024
# ============================================================

download_period \
    "2024" \
    "2024-01-01T00:00:00Z" \
    "2024-12-31T23:59:59Z"


# ============================================================
# 2025
# ============================================================

download_period \
    "2025" \
    "2025-01-01T00:00:00Z" \
    "2025-12-31T23:59:59Z"

# ============================================================
# Finished
# ============================================================

echo
echo "============================================================"
echo "All download periods processed."
echo "Output directory: $BASE_DIR"
echo "============================================================"
