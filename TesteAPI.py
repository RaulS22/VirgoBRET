import requests

api_key = "eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6ImZkYjUwZjMyYzc3YTQ4ZjI4YTUwZTIzYTYzOGM4OTM2IiwiaCI6Im11cm11cjEyOCJ9"

headers = {
    "Authorization": api_key
}

url = "https://api.dataplatform.knmi.nl/open-data/v1/datasets/analog_seismograms/versions/1/files"

r = requests.get(url, headers=headers)

print("STATUS:", r.status_code)
print(r.text)