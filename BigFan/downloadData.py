# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16, 2022

@author: Annalise Miller
"""
import os
from datetime import datetime

import cdsapi
import netCDF4 as nc
import numpy as np
import pandas as pd
import requests


def read_netcdf(path: str, variable_id: str):
    ds = nc.Dataset(path)
    df = pd.DataFrame(
        data=[i[0].filled()[0] for i in ds.variables[variable_id]],
        columns=[variable_id],
        index=ds.variables["time"][:].data,
    )
    df.index = [pd.Timestamp(1900, 1, 1) + pd.Timedelta(hours=i) for i in df.index]
    df.replace(ds.variables[variable_id].missing_value, np.nan, inplace=True)
    return df


def format_ws_wd(df):
    if ("u100" in df.columns) & ("v100" in df.columns):
        df["WS_100m"] = pow(pow(df["u100"], 2) + pow(df["v100"], 2), 0.5)
        df["WD_100m"] = np.arctan(df["v100"] / df["u100"])
        df["WD_100m"][df["WD_100m"] < 180] += 180
        df["WD_100m"][df["WD_100m"] <= 180] -= 180
    if ("u10" in df.columns) & ("v10" in df.columns):
        df["WS_10m"] = pow(pow(df["u10"], 2) + pow(df["v10"], 2), 0.5)
        df["WD_10m"] = np.arctan(df["v10"] / df["u10"])
        df["WD_10m"][df["WD_10m"] < 180] += 180
        df["WD_10m"][df["WD_10m"] <= 180] -= 180
    return df


def pull_era5(
    latitude, longitude, include_10m: bool = False, start_year: int = 2000, end_year: int = None, test: bool = False
):
    if end_year is None:
        end_year = datetime.now().year + 1
    else:
        end_year += 1
    variables = {
        "100m_u_component_of_wind": "u100",
        "100m_v_component_of_wind": "v100",
        "2m_temperature": "t2m",
        "surface_pressure": "sp",
    }
    if include_10m:
        variables["10m_u_component_of_wind"] = "u10"
        variables["10m_v_component_of_wind"] = "v10"
    data_years = [str(data_year) for data_year in range(start_year, end_year)]
    df = pd.DataFrame()
    for variable, variable_id in variables.items():
        df_sub = pd.DataFrame()
        for data_year in data_years:
            api_call = {
                "format": "netcdf",
                "product_type": "reanalysis",
                "variable": [variable],
                "year": data_year,
                "month": ["0" + str(i) if i < 10 else str(i) for i in range(1, 13)],
                "day": ["0" + str(i) if i < 10 else str(i) for i in range(1, 32)],
                "time": ["0" + str(i) + ":00" if i < 10 else str(i) + ":00" for i in range(24)],
                "area": [latitude, longitude, latitude, longitude],  # +  # -  # -  # +
            }
            c = cdsapi.Client()
            c.retrieve("reanalysis-era5-single-levels", api_call, variable + "_" + data_year + ".nc")
            df_sub = pd.concat([df_sub, read_netcdf(variable + "_" + data_year + ".nc", variable_id)])
            # TODO: add file removal
            if test is False:
                try:
                    os.remove(variable + "_" + data_year + ".nc")
                except:
                    print("could not remove file: ", variable + "_" + data_year + ".nc")
        df = pd.concat([df, df_sub], axis=1)
    df = format_ws_wd(df)
    return df


def pull_prism(latitude: float, longitude: float, elevation: float, start_year: bool = None):
    current_date = datetime.now()
    # set time back 6 months to ensure stable version of data
    if current_date.month > 6:
        year = str(current_date.year)
        month = current_date.month - 6
    else:
        year = str(current_date.year - 1)
        month = current_date.month + 6
    if month < 10:
        month = "0" + str(month)
    else:
        month = str(month)
    current_date = year + month + "01"
    if start_year is None:
        start = "1981" + current_date[4:]
    else:
        start = str(start_year) + current_date[4:]
    data = {
        "spares": "4km",
        "interp": "0",
        "stats": ["ppt  tmin    tmean   tmax    tdmean"],
        "units": "si",
        "range": "daily",
        "start": start,
        "end": current_date,
        "stability": "stable",
        "lon": str(longitude),
        "lat": str(latitude),
        "elev": str(elevation),
        "call": "pp/daily_timeseries",
        "proc": "gridserv",
    }
    baseurl = "https://prism.oregonstate.edu/"
    url = baseurl + "explorer/dataexplorer/rpc.php"
    # send API request
    post_response = requests.post(url, data)
    output = post_response.json()
    result = output["result"]
    df = pd.DataFrame.from_dict(result["data"])
    # set dates
    start_date = pd.datetime(int(start[:4]), int(month), 1, 0)
    dates = [start_date + pd.Timedelta(days=i) for i in range(len(df))]
    df.index = dates
    return df


def pull_nasa(latitude: float, longitude: float, elevation: float, start_year: int = None):
    current_date = datetime.now()
    # set time back 6 months to ensure stable version of data
    if current_date.month > 6:
        year = str(current_date.year)
        month = current_date.month - 6
    else:
        year = str(current_date.year - 1)
        month = current_date.month + 6
    if month < 10:
        month = "0" + str(month)
    else:
        month = str(month)
    current_date = year + month + "01"
    if start_year is None:
        start_date = "2001" + current_date[4:]
    elif start_year < 2001:
        raise ValueError("NASA API data is not available prior to 2001")
    else:
        start_date = str(start_year) + current_date[4:]
    data = {
        "parameters": "PRECTOTCORR,T2M,T2MDEW,RH2M,PS,WS50M,WD50M",
        "community": "RE",
        "longitude": str(longitude),
        "latitude": str(latitude),
        "start": start_date,
        "end": current_date,
        "format": "JSON",
        "user": "DAV",
        "site-elevation": str(elevation),
    }
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    # API call
    response = requests.get(url, data)
    output = response.json()
    result = output["properties"]
    df = pd.DataFrame.from_dict(result["parameter"])
    df.index = [datetime.strptime(i, "%Y%m%d%H") for i in df.index]
    return df


if __name__ == "__main__":
    # my_df = pull_era5(45, -120, start_year=2015, end_year=2018, test=True)
    # my_df.to_csv(r'C:\Users\Annalise\Documents\BigFan\tests\era5_2019.csv')
    test_lat = 45
    test_lon = -120
    test_elevation = 1352
    nasa_df = pull_nasa(test_lat, test_lon, test_elevation, start_year=2018)
    # prism_df = pull_prism(test_lat, test_lon, test_elevation)
    nasa_df.to_csv(r"C:\Users\Annalise\Downloads\nasaTest.csv")
