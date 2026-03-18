"""
Downloader for KNMI Open Data datasets
https://developer.dataplatform.knmi.nl/open-data-api
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
#from typing import Any, Tuple, List, Dict

import requests
from requests import Session

#"https://api.dataplatform.knmi.nl/open-data/v1" analog_seismograms "1"
#"https://dataplatform.knmi.nl/dataset/aardbevingen-cijfers-1" "aardbevingen_cijfers" "1"
# "operational_netherlands_earthquake_magnitude_completeness" "1.0"
# "seismic_shakemaps" "2.0"
# "rdsa_inventory_changelog" "2.0"
# "netherlands_earthquake_location_uncertainty" "1.0"
# "netherlands_earthquake_magnitude_completeness" "1.0"
# https://api.dataplatform.knmi.nl/open-data/v1/datasets/seismic_hazardmaps/versions/1/files

# https://dataplatform.knmi.nl/dataset/nam-epos-dataset-1-0 "nam_epos_dataset" "1.0" # ler: https://www.nam.nl/english-information.html 
# 500GB um (1) arquivo


# -----------------------------
# CONFIGURATION
# -----------------------------

API_KEY = "eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6ImZkYjUwZjMyYzc3YTQ4ZjI4YTUwZTIzYTYzOGM4OTM2IiwiaCI6Im11cm11cjEyOCJ9"

BASE_URL = "https://api.dataplatform.knmi.nl/open-data/v1"
DATASET_NAME = "seismic_hazardmaps" 
DATASET_VERSION = "1" #"1" "1.0" "2.0"

DOWNLOAD_DIRECTORY = "./dataset-download"


MAX_KEYS = 100
OVERWRITE = False



# -----------------------------
# LOGGING
# -----------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# API FUNCTIONS
# -----------------------------

def list_dataset_files(
    session: Session,
    base_url: str,
    dataset_name: str,
    dataset_version: str,
    params: dict[str, str],
) -> tuple[list[str], dict[str, any]]:

    endpoint = f"{base_url}/datasets/{dataset_name}/versions/{dataset_version}/files"

    response = session.get(endpoint, params=params)

    if response.status_code != 200:
        raise Exception(f"Unable to list dataset files: {response.text}")

    data = response.json()

    filenames = [file["filename"] for file in data["files"]]

    return filenames, data


def get_download_url(
    session: Session,
    base_url: str,
    dataset_name: str,
    dataset_version: str,
    filename: str,
) -> str:

    endpoint = (
        f"{base_url}/datasets/{dataset_name}/versions/{dataset_version}"
        f"/files/{filename}/url"
    )

    r = session.get(endpoint)

    if r.status_code != 200:
        raise Exception(f"Unable to get download URL for {filename}")

    return r.json()["temporaryDownloadUrl"]


# -----------------------------
# DOWNLOAD FUNCTIONS
# -----------------------------

def download_file(download_url: str, directory: str, filename: str) -> tuple[bool, str]:

    filepath = Path(directory, filename)

    try:

        with requests.get(download_url, stream=True) as r:

            r.raise_for_status()

            with open(filepath, "wb") as f:

                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        logger.info(f"Downloaded {filename}")

        return True, filename

    except Exception:

        logger.exception(f"Failed downloading {filename}")

        return False, filename


def download_dataset_file(
    session: Session,
    base_url: str,
    dataset_name: str,
    dataset_version: str,
    filename: str,
    directory: str,
    overwrite: bool,
) -> tuple[bool, str]:

    filepath = Path(directory, filename)

    if filepath.exists() and not overwrite:
        logger.info(f"Skipping existing file {filename}")
        return True, filename

    try:

        download_url = get_download_url(
            session,
            base_url,
            dataset_name,
            dataset_version,
            filename,
        )

        return download_file(download_url, directory, filename)

    except Exception:

        logger.exception(f"Failed processing {filename}")

        return False, filename


# -----------------------------
# THREAD CONTROL
# -----------------------------

def get_max_worker_count(filesizes: list[int]) -> int:

    size_for_threading = 10_000_000  # 10 MB

    avg = sum(filesizes) / len(filesizes)

    return 1 if avg > size_for_threading else 10


# -----------------------------
# MAIN
# -----------------------------

async def main():

    Path(DOWNLOAD_DIRECTORY).mkdir(exist_ok=True)

    session = requests.Session()
    session.headers.update({"Authorization": API_KEY})

    filenames: list[str] = []
    file_sizes: list[int] = []

    next_page_token = None

    while True:

        params = {"maxKeys": str(MAX_KEYS)}

        if next_page_token:
            params["nextPageToken"] = next_page_token

        dataset_filenames, response_json = list_dataset_files(
            session,
            BASE_URL,
            DATASET_NAME,
            DATASET_VERSION,
            params,
        )

        filenames.extend(dataset_filenames)

        file_sizes.extend(file["size"] for file in response_json["files"])

        next_page_token = response_json.get("nextPageToken")

        if not response_json.get("isTruncated"):
            break

    logger.info(f"Total files to download: {len(filenames)}")

    worker_count = get_max_worker_count(file_sizes)

    executor = ThreadPoolExecutor(max_workers=worker_count)

    loop = asyncio.get_event_loop()

    tasks = [

        loop.run_in_executor(
            executor,
            download_dataset_file,
            session,
            BASE_URL,
            DATASET_NAME,
            DATASET_VERSION,
            filename,
            DOWNLOAD_DIRECTORY,
            OVERWRITE,
        )

        for filename in filenames
    ]

    results = await asyncio.gather(*tasks)

    failures = [name for success, name in results if not success]

    if failures:

        logger.warning("Failed downloads:")
        for f in failures:
            logger.warning(f)

    logger.info("Download finished")


# -----------------------------
# ENTRYPOINT
# -----------------------------

if __name__ == "__main__":

    asyncio.run(main())