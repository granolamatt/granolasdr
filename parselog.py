import numpy as np
import datetime
import simplekml

def to_location(maiden: str, center: bool = False) -> tuple[float, float]:
    """
    convert Maidenhead grid to latitude, longitude

    Parameters
    ----------

    maiden : str
        Maidenhead grid locator of length 2 to 8

    center : bool
        If true, return the center of provided maidenhead grid square, instead of default south-west corner
        Default value = False needed to maidenhead full backward compatibility of this module.

    Returns
    -------

    latLon : tuple of float
        Geographic latitude, longitude
    """

    maiden = maiden.strip().upper()

    N = len(maiden)
    if not ((8 >= N >= 2) and (N % 2 == 0)):
        raise ValueError("Maidenhead locator requires 2-8 characters, even number of characters")

    Oa = ord("A")
    lon = -180.0
    lat = -90.0
    # %% first pair
    lon += (ord(maiden[0]) - Oa) * 20
    lat += (ord(maiden[1]) - Oa) * 10
    # %% second pair
    if N >= 4:
        lon += int(maiden[2]) * 2
        lat += int(maiden[3]) * 1
    # %%
    if N >= 6:
        lon += (ord(maiden[4]) - Oa) * 5.0 / 60
        lat += (ord(maiden[5]) - Oa) * 2.5 / 60
    # %%
    if N >= 8:
        lon += int(maiden[6]) * 5.0 / 600
        lat += int(maiden[7]) * 2.5 / 600

    # %% move lat lon to the center (if requested)
    if center:
        if N == 2:
            lon += 20 / 2
            lat += 10 / 2
        elif N == 4:
            lon += 2 / 2
            lat += 1.0 / 2
        elif N == 6:
            lon += 5.0 / 60 / 2
            lat += 2.5 / 60 / 2
        elif N >= 8:
            lon += 5.0 / 600 / 2
            lat += 2.5 / 600 / 2

    return lat, lon
def convertFreq(bin):
    freq_conv = [{"start":1.8,"stop":2.0,"binstart":0,"binstop":31992.1875},
    {"start":3.5,"stop":4.0,"binstart":31992.1875,"binstop":111930.0},
    {"start":5.3305,"stop":5.4355,"binstart":111930.0,"binstop":128821.875},
    {"start":7.0,"stop":7.3,"binstart":128821.875,"binstop":176852.8125},
    {"start":10.1,"stop":10.15,"binstart":176852.8125,"binstop":184957.5},
    {"start":14.0,"stop":14.35,"binstart":184957.5,"binstop":240922.5},
    {"start":18.068,"stop":18.168,"binstart":240922.5,"binstop":256961.25},
    {"start":21.0,"stop":21.45,"binstart":256961.25,"binstop":328965.0},
    {"start":24.89,"stop":24.99,"binstart":328965.0,"binstop":345003.75},
    {"start":28,"stop":29.7,"binstart":345003.75,"binstop":616638.75}]

    for dd in freq_conv:
        if dd['binstart'] < bin and dd['binstop'] > bin:
            delta = (bin - dd['binstart']) * 6.2600160256410255 / 1e6
            return dd['start'] + delta


with open('monitor.txt') as ff:
    lines = ff.readlines()

coords = []
for line in lines:
    if line[0] == '+' or line[0] == '-':
        try:
            parts = line.split(' ')
            ts = datetime.datetime.fromtimestamp(float(parts[1][1:]))
            freqbin = convertFreq(float(parts[3]))
            freqbin = np.round(freqbin, decimals=6)
            message = parts[-1][:-1]
            if len(message) == 4:
                lat,lon = to_location(message, center=True)
                print(ts, "freq", freqbin,"lat", lat, "long", lon)
                coords.append([str(freqbin),str(lat),str(lon)])
        except:
            print("Bad value")

kml = simplekml.Kml()
for desc,lat,lon in coords:
    kml.newpoint(description=desc,
        coords=[(lon, lat)])  # lon, lat, optional height
# save KML to a file
kml.save("freqs.kml")
