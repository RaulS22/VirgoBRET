from obspy import read, read_inventory

mseed_file = "/home/rauls/Desktop/VirgoBRET/SENA-files/2022/eida_response_MN-SENA_20220101000000_20220131235959.mseed"
inventory_file = "fdsn_station.xml"

output_file = "debug_response.txt"

st = read(mseed_file)
inv = read_inventory(inventory_file)
print(inv)

with open(output_file, "w") as f:

    def write(msg):
        f.write(msg + "\n")

    write("========== TRACE INFO ==========")
    for tr in st:
        write(f"\nTrace ID: {tr.id}")
        write(f"Network : {tr.stats.network}")
        write(f"Station : {tr.stats.station}")
        write(f"Location: '{tr.stats.location}'")
        write(f"Channel : {tr.stats.channel}")
        write(f"Start   : {tr.stats.starttime}")
        write(f"End     : {tr.stats.endtime}")

    write("\n========== INVENTORY CHANNELS ==========")
    for net in inv:
        for sta in net:
            for cha in sta:
                write(f"{net.code}.{sta.code}.{cha.location_code}.{cha.code} "
                      f"| {cha.start_date} → {cha.end_date}")

    write("\n========== MATCH TEST ==========")
    for tr in st:
        write(f"\nChecking: {tr.id}")

        try:
            inv.get_response(tr.id, tr.stats.starttime)
            write("✔ Response FOUND")

        except Exception:
            write("✘ Response NOT found")

            write("---- TRACE ----")
            write(tr.id)

            write("---- POSSIBLE MATCHES ----")
            for net in inv:
                for sta in net:
                    if sta.code != tr.stats.station:
                        continue
                    for cha in sta:
                        write(f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}")

            write("---- HINTS ----")
            write("- location code mismatch ('' vs '00')")
            write("- channel mismatch (HHZ vs BHZ etc.)")
            write("- network mismatch")
            write("- time coverage mismatch")

print(f"Debug output saved to: {output_file}")
