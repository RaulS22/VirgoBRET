from __future__ import annotations

import requests
import pandas as pd

from pathlib import Path

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn import RoutingClient
from obspy.clients.fdsn.header import FDSNNoDataException


# ============================================================
# CONFIGURATION
# ============================================================

YEARS = [2022, 2023, 2024, 2025]

OUTPUT_DIR = Path("virgo_eida_check")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# INGV FDSN Station service
INGV_URL = "https://webservices.ingv.it"

# EIDA Availability service
EIDA_AVAILABILITY_URL = (
    "https://www.orfeus-eu.org/eidaws/availability/1/query"
)

REQUEST_TIMEOUT = 60


# ============================================================
# CANDIDATES
# ============================================================
#
# For now, VR is the important candidate.
#
# OX.VR is kept here as a historical fallback, but the INGV
# endpoint may not contain this representation.
#

CANDIDATES = [
    {"network": "VR", "station": "*"},
    {"network": "OX", "station": "VR"},
]


# ============================================================
# HELPERS
# ============================================================

def utc_year_start(year: int) -> UTCDateTime:
    return UTCDateTime(f"{year}-01-01T00:00:00")


def utc_year_end(year: int) -> UTCDateTime:
    return UTCDateTime(f"{year + 1}-01-01T00:00:00")


def normalize_location(location: str | None) -> str:
    """
    Convert blank FDSN location to an empty string.
    """

    if location is None:
        return ""

    if location.strip() == "":
        return ""

    return location.strip()


def eida_location(location: str) -> str:
    """
    FDSN/EIDA representation of an empty location.

    Empty location is represented by '--' in Availability queries.
    """

    if location.strip() == "":
        return "--"

    return location


def datetime_to_string(value) -> str | None:
    """
    Convert ObsPy UTCDateTime to an ISO string.

    This is important because pandas may have problems hashing
    UTCDateTime objects during drop_duplicates().
    """

    if value is None:
        return None

    try:
        return value.isoformat()

    except Exception:
        return str(value)


def overlap_seconds(
    start_a: UTCDateTime,
    end_a: UTCDateTime,
    start_b: UTCDateTime,
    end_b: UTCDateTime,
) -> float:
    """
    Return overlap duration between two time intervals in seconds.
    """

    start = max(start_a, start_b)
    end = min(end_a, end_b)

    return max(0.0, end - start)


# ============================================================
# 1. DISCOVER CHANNELS USING INGV FDSN STATION
# ============================================================

def query_ingv_candidate(
    network: str,
    station: str,
) -> list[dict]:

    print()
    print("=" * 80)
    print(f"Querying INGV Station Service: {network}.{station}")
    print("=" * 80)

    client = Client(
        INGV_URL,
        timeout=REQUEST_TIMEOUT,
    )

    try:

        inventory = client.get_stations(
            network=network,
            station=station,
            level="channel",
            includerestricted=False,
        )

    except FDSNNoDataException:

        print(
            f"No Station data returned for "
            f"{network}.{station}"
        )

        return []

    except Exception as exc:

        print(
            f"ERROR while querying "
            f"{network}.{station}:"
        )

        print(exc)

        return []

    results = []

    for net in inventory:

        for sta in net:

            for cha in sta.channels:

                location = normalize_location(
                    cha.location_code
                )

                results.append(
                    {
                        "network": net.code,
                        "station": sta.code,
                        "location": location,
                        "channel": cha.code,

                        "station_name": (
                            sta.site.name
                            if sta.site is not None
                            else ""
                        ),

                        "latitude": sta.latitude,
                        "longitude": sta.longitude,
                        "elevation_m": sta.elevation,

                        # Convert UTCDateTime to strings!
                        "channel_start": datetime_to_string(
                            cha.start_date
                        ),

                        "channel_end": datetime_to_string(
                            cha.end_date
                        ),

                        "sample_rate_hz": cha.sample_rate,
                    }
                )

    print(
        f"Found {len(results)} channel entries."
    )

    return results


# ============================================================
# 2. OPTIONAL EIDA ROUTING SEARCH
# ============================================================

def query_eida_routing(
    network: str,
    station: str,
) -> list[dict]:

    print()
    print("=" * 80)
    print(f"Trying EIDA Routing Service: {network}.{station}")
    print("=" * 80)

    try:

        routing_client = RoutingClient(
            "eida-routing"
        )

        inventory = routing_client.get_stations(
            network=network,
            station=station,
            level="channel",
            includerestricted=False,
        )

    except Exception as exc:

        print(
            "EIDA routing query failed:"
        )

        print(exc)

        return []

    results = []

    for net in inventory:

        for sta in net:

            for cha in sta.channels:

                location = normalize_location(
                    cha.location_code
                )

                results.append(
                    {
                        "network": net.code,
                        "station": sta.code,
                        "location": location,
                        "channel": cha.code,

                        "station_name": (
                            sta.site.name
                            if sta.site is not None
                            else ""
                        ),

                        "latitude": sta.latitude,
                        "longitude": sta.longitude,
                        "elevation_m": sta.elevation,

                        "channel_start": datetime_to_string(
                            cha.start_date
                        ),

                        "channel_end": datetime_to_string(
                            cha.end_date
                        ),

                        "sample_rate_hz": cha.sample_rate,
                    }
                )

    print(
        f"Found {len(results)} channel entries."
    )

    return results


# ============================================================
# 3. QUERY AVAILABILITY
# ============================================================

def query_availability(
    network: str,
    station: str,
    location: str,
    channel: str,
    start: UTCDateTime,
    end: UTCDateTime,
) -> list[dict]:

    params = {
        "network": network,
        "station": station,
        "location": eida_location(location),
        "channel": channel,

        "starttime": start.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),

        "endtime": end.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),

        "merge": "overlap",
        "mergegaps": 86400,

        "format": "text",
        "nodata": 404,
    }

    try:

        response = requests.get(
            EIDA_AVAILABILITY_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

    except requests.RequestException as exc:

        return [
            {
                "status": "REQUEST_ERROR",
                "error": str(exc),
            }
        ]

    # --------------------------------------------------------
    # No data
    # --------------------------------------------------------

    if response.status_code in (204, 404):

        return [
            {
                "status": "NO_DATA"
            }
        ]

    # --------------------------------------------------------
    # HTTP error
    # --------------------------------------------------------

    if response.status_code != 200:

        return [
            {
                "status": "HTTP_ERROR",
                "error": (
                    f"HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                ),
            }
        ]

    # --------------------------------------------------------
    # Parse response
    # --------------------------------------------------------

    lines = response.text.splitlines()

    data_lines = []

    for line in lines:

        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("#"):
            continue

        data_lines.append(stripped)

    if not data_lines:

        return [
            {
                "status": "NO_DATA"
            }
        ]

    results = []

    for line in data_lines:

        fields = line.split()

        if len(fields) < 6:
            continue

        net = fields[0]
        sta = fields[1]
        loc = fields[2]
        cha = fields[3]

        start_time = fields[4]
        end_time = fields[5]

        try:

            span_start = UTCDateTime(
                start_time
            )

            span_end = UTCDateTime(
                end_time
            )

        except Exception:

            continue

        results.append(
            {
                "status": "AVAILABLE",

                "network": net,
                "station": sta,
                "location": loc,
                "channel": cha,

                "available_start": span_start,
                "available_end": span_end,
            }
        )

    if not results:

        return [
            {
                "status": "NO_DATA"
            }
        ]

    return results


# ============================================================
# 4. CALCULATE YEARLY AVAILABILITY
# ============================================================

def calculate_yearly_availability(
    channel_info: dict,
) -> list[dict]:

    network = channel_info["network"]
    station = channel_info["station"]
    location = channel_info["location"]
    channel = channel_info["channel"]

    results = []

    for year in YEARS:

        year_start = utc_year_start(year)
        year_end = utc_year_end(year)

        print(
            f"Checking "
            f"{network}.{station}."
            f"{location or '--'}."
            f"{channel} "
            f"{year}..."
        )

        availability = query_availability(
            network=network,
            station=station,
            location=location,
            channel=channel,
            start=year_start,
            end=year_end,
        )

        available_seconds = 0.0
        intervals = 0
        errors = []

        for item in availability:

            status = item.get("status")

            if status == "AVAILABLE":

                intervals += 1

                available_seconds += (
                    overlap_seconds(
                        item["available_start"],
                        item["available_end"],
                        year_start,
                        year_end,
                    )
                )

            elif status not in (
                "NO_DATA",
                None,
            ):

                errors.append(
                    item.get(
                        "error",
                        status
                    )
                )

        total_seconds = (
            year_end - year_start
        )

        coverage_percent = (
            100.0
            * available_seconds
            / total_seconds
        )

        results.append(
            {
                "network": network,
                "station": station,
                "location": location,
                "channel": channel,

                "year": year,

                "available_seconds":
                    available_seconds,

                "available_hours":
                    available_seconds / 3600.0,

                "coverage_percent":
                    coverage_percent,

                "interval_count":
                    intervals,

                "channel_start":
                    channel_info[
                        "channel_start"
                    ],

                "channel_end":
                    channel_info[
                        "channel_end"
                    ],

                "sample_rate_hz":
                    channel_info[
                        "sample_rate_hz"
                    ],

                "latitude":
                    channel_info[
                        "latitude"
                    ],

                "longitude":
                    channel_info[
                        "longitude"
                    ],

                "elevation_m":
                    channel_info[
                        "elevation_m"
                    ],

                "station_name":
                    channel_info[
                        "station_name"
                    ],

                "error":
                    " | ".join(errors),
            }
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("VIRGO SEISMIC DATA DISCOVERY")
    print("EIDA / INGV / FDSN")
    print("=" * 80)
    print()

    all_channels = []

    # ========================================================
    # STEP 1
    # Query candidate networks
    # ========================================================

    for candidate in CANDIDATES:

        results = query_ingv_candidate(
            network=candidate["network"],
            station=candidate["station"],
        )

        all_channels.extend(
            results
        )

    # ========================================================
    # STEP 2
    # Construct DataFrame
    # ========================================================

    if all_channels:

        channels_df = pd.DataFrame(
            all_channels
        )

        # All date fields are strings now,
        # so drop_duplicates() is safe.

        channels_df = (
            channels_df
            .drop_duplicates(
                subset=[
                    "network",
                    "station",
                    "location",
                    "channel",
                    "channel_start",
                    "channel_end",
                ]
            )
            .reset_index(drop=True)
        )

    else:

        channels_df = pd.DataFrame()

    # ========================================================
    # STEP 3
    # If nothing was found, try EIDA Routing
    # ========================================================

    if channels_df.empty:

        print()
        print("=" * 80)
        print(
            "Direct INGV search returned no channels."
        )
        print(
            "Trying EIDA Routing Service..."
        )
        print("=" * 80)

        routing_results = []

        for candidate in CANDIDATES:

            results = query_eida_routing(
                network=candidate["network"],
                station=candidate["station"],
            )

            routing_results.extend(
                results
            )

        if routing_results:

            channels_df = pd.DataFrame(
                routing_results
            )

            channels_df = (
                channels_df
                .drop_duplicates(
                    subset=[
                        "network",
                        "station",
                        "location",
                        "channel",
                        "channel_start",
                        "channel_end",
                    ]
                )
                .reset_index(drop=True)
            )

    # ========================================================
    # STEP 4
    # No channels
    # ========================================================

    if channels_df.empty:

        print()
        print("=" * 80)
        print("NO VIRGO SEISMIC CHANNELS FOUND")
        print("=" * 80)
        print()

        print(
            "The FDSN services queried did not return "
            "a station/channel matching the candidates."
        )

        print()
        print(
            "This does NOT necessarily mean that Virgo "
            "has no seismic data."
        )

        return

    # ========================================================
    # STEP 5
    # Show discovered channels
    # ========================================================

    print()
    print("=" * 80)
    print("CHANNELS FOUND BY INGV / EIDA")
    print("=" * 80)

    display_columns = [
        "network",
        "station",
        "location",
        "channel",
        "sample_rate_hz",
        "channel_start",
        "channel_end",
    ]

    print(
        channels_df[
            display_columns
        ].to_string(index=False)
    )

    # ========================================================
    # STEP 6
    # Save channel inventory
    # ========================================================

    channels_csv = (
        OUTPUT_DIR /
        "virgo_channels.csv"
    )

    channels_df.to_csv(
        channels_csv,
        index=False,
    )

    print()
    print(
        f"Channel inventory saved to:\n"
        f"{channels_csv}"
    )

    # ========================================================
    # STEP 7
    # Calculate yearly availability
    # ========================================================

    availability_results = []

    for _, channel_info in channels_df.iterrows():

        channel_dict = (
            channel_info.to_dict()
        )

        results = (
            calculate_yearly_availability(
                channel_dict
            )
        )

        availability_results.extend(
            results
        )

    if not availability_results:

        print()
        print(
            "No availability results were generated."
        )

        return

    availability_df = pd.DataFrame(
        availability_results
    )

    # ========================================================
    # STEP 8
    # Save complete availability table
    # ========================================================

    availability_csv = (
        OUTPUT_DIR /
        "virgo_availability_2022_2025.csv"
    )

    availability_df.to_csv(
        availability_csv,
        index=False,
    )

    print()
    print(
        f"Availability results saved to:\n"
        f"{availability_csv}"
    )

    # ========================================================
    # STEP 9
    # Print summary
    # ========================================================

    print()
    print("=" * 80)
    print("YEARLY AVAILABILITY SUMMARY")
    print("=" * 80)

    summary_columns = [
        "network",
        "station",
        "location",
        "channel",
        "year",
        "coverage_percent",
        "available_hours",
        "interval_count",
    ]

    print(
        availability_df[
            summary_columns
        ].to_string(index=False)
    )

    # ========================================================
    # STEP 10
    # Create year × channel matrix
    # ========================================================

    pivot = (
        availability_df
        .pivot_table(
            index=[
                "network",
                "station",
                "location",
                "channel",
            ],
            columns="year",
            values="coverage_percent",
            aggfunc="first",
        )
    )

    pivot_csv = (
        OUTPUT_DIR /
        "virgo_availability_matrix.csv"
    )

    pivot.to_csv(
        pivot_csv
    )

    print()
    print("=" * 80)
    print("YEARLY COVERAGE MATRIX (%)")
    print("=" * 80)

    print(
        pivot.to_string()
    )

    print()
    print(
        f"Coverage matrix saved to:\n"
        f"{pivot_csv}"
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()