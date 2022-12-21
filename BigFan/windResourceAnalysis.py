# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15, 2022

@author: Annalise Miller
"""

import os
from datetime import datetime
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata
from utm import from_latlon

from BigFan import mcp

my_colors = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#ffff99",
    "#b15928",
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
]


def check_combined_data(masts: list, are_masts: bool = True) -> list:
    """
    Check input data for common issues

    Parameters
    ----------
    masts : list
        list of Mast objects to check
    are_masts : bool, optional
        Whether the input data represent mast measurements (True) or reanalysis data (False). The default is True.

    Returns
    -------
    list
        list of warnings about the input data.

    """
    warnings = []
    direction = pd.DataFrame()
    ids = []
    utm_zones = []
    for mast in masts:
        ids.append(mast.id)
        direction[mast.id] = mast.MastData.df[mast.MastData.primary_windDirection_column] * 1
        direction[mast.id][(direction[mast.id] < 60) | (direction[mast.id] > 360)] = np.nan
        utm_zones.append(mast.utm_zone)
    if len(set(utm_zones)) > 1:
        utm_zone = min(utm_zones)
        for mast in masts:
            if mast.utm_zone > utm_zone:
                mast.utm_zone = utm_zone
                easting, northing, zone, letter = from_latlon(mast.lat, mast.lon, force_zone_number=utm_zone)
                mast.easting = easting
                mast.northing = northing
    direction.dropna(inplace=True)
    dir_corr = direction.corr()
    slopes = pd.DataFrame(index=ids, columns=ids)
    offsets = pd.DataFrame(index=ids, columns=ids)
    for mast1 in my_masts:
        for mast2 in my_masts:
            if mast1.id != mast2.id:
                slope, offset = mcp.correlate_lls(direction, mast1.id, mast2.id)
                slopes[mast1.id][mast2.id] = slope
                offsets[mast1.id][mast2.id] = offset
    if are_masts:
        if dir_corr.min().min() < 0.9:
            warnings.append(
                "Correlation coefficient between the primary direction sensor on each mast is < 0.9. This may indicate "
                + "a time shift in the input data. Please check."
            )
        if offsets.abs().max().max() > 20:
            warnings.append(
                "The offset from the correlation between the primary direction sensor on each mast is > 20. This may "
                + "indicate a miscalibration in one or more direction sector. Please check."
            )
    else:
        if dir_corr.min().min() < 0.9:
            warnings.append(
                "Correlation coefficient between the primary direction sensor on each long-term source is < 0.9. This "
                + "may indicate a time shift in the input data. Please check."
            )
    return warnings


def apply_global_pc(time_series: pd.Series, p_curve: pd.DataFrame, prod_col: str):
    """
    Run a time series of wind speeds through a power curves

    Parameters
    ----------
    time_series : pd.Series
        time series of wind speeds (float)
    p_curve : pd.DataFrame
        global power curve
    prod_col : str
        name of wind speed column in power curve data frame

    Returns
    -------
    time_series : np.array
        wind speed and production time series
    """
    return np.interp(time_series, p_curve.index, p_curve[prod_col])


def long_term_correlations(masts: list, lt_sources: list, corr_type: Callable, lt_dur: str = "W", mast_dur: str = "D"):
    """
    Correlate between onsite masts and reference data sources

    Parameters
    ----------
    masts : list
        list of Mast objects representing on-site data.
    lt_sources : list
        list of Mast objects representing long-term data.
    corr_type : Callable
        correlation type, may be correlate_TLS, or correlate_LLS.
        TLS: total least squares
        LLS: linear least squares
    lt_dur : str, optional
        time scale to correlate between masts and long-term sources. The default is 'W'.
    mast_dur : str, optional
        time scale to correlate between on-site masts. The default is 'D'.

    Returns
    -------
    None. Does set each masts' lt_corr and mast_corr attributes with dataframes
    including correlation data points, slope, offset, and correlation coefficient

    """
    for mast in masts:
        for lt in lt_sources:
            # setup correlation
            sub_df = pd.DataFrame()
            sub_df["y"] = mast.MastData.df[mast.MastData.primary_windSpeed_column]
            sub_df["x"] = lt.MastData.df[lt.MastData.primary_windSpeed_column]
            sub_df = sub_df.resample(lt_dur).mean()
            sub_df.dropna(inplace=True)
            # time steps
            steps = len(sub_df)
            # run correlation r2
            r2 = sub_df["x"].corr(sub_df["y"])
            # run slope & offset
            slope, offset = corr_type(sub_df, "x", "y")
            # save data to mast
            mast.MastData.longTerm_corr[lt.id] = [steps, offset, slope, r2]
        for mast2 in masts:
            if mast != mast2:
                # setup correlation
                sub_df = pd.DataFrame()
                sub_df["y"] = mast2.MastData.df[mast2.MastData.primary_windSpeed_column]
                sub_df["x"] = mast.MastData.df[mast.MastData.primary_windSpeed_column]
                sub_df = sub_df.resample(mast_dur).mean()
                sub_df.dropna(inplace=True)
                # time steps
                steps = len(sub_df)
                # run correlation r2
                r2 = sub_df["x"].corr(sub_df["y"])
                # run slope & offset
                slope, offset = corr_type(sub_df, "x", "y")
                # save data to mast
                mast.MastData.mast_corr[mast2.id] = [steps, offset, slope, r2]
            else:
                mast.MastData.mast_corr[mast2.id] = [
                    len(mast2.MastData.df[mast.MastData.primary_windSpeed_column].resample(mast_dur).mean().dropna()),
                    0,
                    1,
                    1,
                ]


def create_heat_map(df: pd.DataFrame):
    """
    Create a heat map of the site

    Parameters
    ----------
    df : pd.DataFrame
        dataframe including northing, easting, and wind speed columns.
        Will use northing and easting for point locations and wind speed column
        for heatmap coloration

    Returns
    -------
    figPath : str
        filepath to the heatmap figure created

    """
    x_mean = (df["easting"].max() - df["easting"].min()) / 2 + df["easting"].min()
    y_mean = (df["northing"].max() - df["northing"].min()) / 2 + df["northing"].min()
    buffer = (
        max([(df["easting"].max() - df["easting"].min()), (df["northing"].max() - df["northing"].min())]) / 2 * 1.03
    )
    plt.plot(figsize=(6.4, 6.4))
    plt.xlim([x_mean - buffer, x_mean + buffer])
    plt.ylim([y_mean - buffer, y_mean + buffer])
    nx, ny = np.mgrid[x_mean - buffer: x_mean + buffer: 1000, y_mean - buffer: y_mean + buffer: 1000]
    h = [(i["easting"], i["northing"]) for ct, i in df.iterrows()]
    nz = griddata(h, df["wind speed"].values, (nx, ny), method="linear")
    plt.contour(nx, ny, nz, linewidths=0.5)
    plt.contourf(nx, ny, nz)
    plt.scatter(df["easting"], df["northing"], s=7, c="k")
    for ind, label in df.iterrows():
        plt.annotate(ind, (label["easting"] + buffer * 0.03, label["northing"] + buffer * 0.03))
    fig_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "figs", "WSHeatMap.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def run_concurrent_period(
    masts: list, power_curve: pd.DataFrame, prod_col: str = "prod", ref_mast: Optional[str] = None
) -> (pd.DataFrame, int, list):
    """
    Return summary of mast concurrent period

    Parameters
    ----------
    masts : list
        list of Mast objects.
    power_curve : pd.DataFrame
        data frame representing turbine power curve
    prod_col : str
        name of column in powerCurve data frame representing power. The default is 'prod'
    ref_mast : str
        mast id of the selected reference mast. The default is None

    Returns
    -------
    df_concurrentPeriod, max_length, fig : tuple(pd.DataFrame, int, list[str])
        dataframe containing summary of concurrent period analysis
        maximum length of a gap filled data set without data extension (for use in mcp process)
        list of paths to figures generated in the concurrent process

    """
    rated_power = power_curve[prod_col].max()
    # set up concurrent period results table
    df_concurrent_period = pd.DataFrame(index=[i.id for i in masts])
    df_concurrent_period["Measurement Height (m)"] = [i.MastData.primary_windSpeed_column_ht for i in masts]
    # compile concurrent wind speed data
    df_heat_map = pd.DataFrame(
        {
            "northing": [i.northing for i in masts],
            "easting": [i.easting for i in masts],
            "wind speed": [np.nan for _ in masts],
        },
        index=[i.id for i in masts],
    )
    cp = pd.DataFrame(
        [i.MastData.df[i.MastData.primary_windSpeed_column] for i in masts], index=[i.id for i in masts]
    ).T
    max_length_df = len(cp.dropna(how="all"))
    cp.dropna(inplace=True)
    for mast in masts:
        mast.MastData.df["CP"] = np.nan
        for ind in cp.index:
            if ind in mast.MastData.df.index:
                mast.MastData.df["CP"][ind] = 1
    df_heat_map["wind speed"] = cp.mean()
    fig = create_heat_map(df_heat_map)
    # save concurrent period wind speeds
    length_concurrent_period = len(cp)
    df_concurrent_period["valid time steps"] = length_concurrent_period
    df_concurrent_period["mean wind speed"] = cp.mean().T
    if ref_mast is not None:
        fig = [fig] + [os.path.join(os.path.abspath(os.path.dirname(__file__)), "figs", "WSRatio.png")]
        # save values for energy content comparison
        scale = df_concurrent_period["mean wind speed"][ref_mast] / df_concurrent_period["mean wind speed"]
        plt.plot(range(len(scale)), 1 / scale, color=my_colors[0])
        plt.axhline(y=1.0, linestyle=":", color="k")
        plt.xticks(range(len(scale)), cp.columns)
        plt.ylabel("Wind speed ratio relative to " + ref_mast)
        plt.savefig(fig[-1])
        plt.close()
    # initiate next columns
    df_concurrent_period["zero power"] = np.nan
    df_concurrent_period["rated power"] = np.nan
    df_concurrent_period["AEP"] = np.nan
    df_concurrent_period["NCF"] = np.nan
    # noinspection PyUnboundLocalVariable
    df_concurrent_period["scaled wind speed"] = df_concurrent_period["mean wind speed"] * scale
    df_concurrent_period["scaled AEP"] = np.nan
    df_concurrent_period["scaled NCF"] = np.nan
    for mast in cp:
        # use power curve to convert wind speeds to production
        production = apply_global_pc(cp[mast], power_curve, prod_col)
        # count non-production
        df_concurrent_period["zero power"][mast] = production[production == 0].size / length_concurrent_period
        # count rated production
        df_concurrent_period["rated power"][mast] = (
            production[production == rated_power].size / length_concurrent_period
        )
        # save production statistics
        df_concurrent_period["AEP"][mast] = production.sum()
        df_concurrent_period["NCF"][mast] = df_concurrent_period["AEP"][mast] / (rated_power * length_concurrent_period)
        # scale wind speeds and reapply power curve for energy content comparison
        production = apply_global_pc(cp[mast] * scale[mast], power_curve, prod_col)
        df_concurrent_period["scaled AEP"][mast] = production.sum()
        df_concurrent_period["scaled NCF"][mast] = df_concurrent_period["scaled AEP"][mast] / (
            rated_power * length_concurrent_period
        )
    return df_concurrent_period, max_length_df, fig


def get_reference_periods(masts: list):
    """
    create a summary table of possible reference periods

    Parameters
    ----------
    masts : list
        list of Mast objects containing MastData.

    Returns
    -------
    reference period options : pd.DataFrame
        data frame summarizing possible reference periods on the pareto front
        representing highest data recovery and lowest mean bias.
    default_period : int
        index of the reference period options data frame that maximizes the number
        of years of data while maintaining a data recovery rate within 3% of the
        maximum data recovery rate in the reference period options

    """
    # select possible reference periods
    earliest_start = min([mast.MastData.data_start for mast in masts])
    latest_end = max([mast.MastData.data_end for mast in masts])

    # get starting point for ref period options
    ref_year = earliest_start.year
    ref_month = earliest_start.month
    if earliest_start.day > 1:
        ref_month += 1

    # determine the maximum number of years and months to search
    max_years = latest_end.year - earliest_start.year
    # month_options = latest_end.month - earliest_start.month + 1
    if latest_end.month < earliest_start.month:
        max_years -= 1
        # month_options += 12

    # determine reference periods to analyze
    ref_period_options = pd.DataFrame(index=["start", "end", "possible data points", "length", "recovery rate", "MAE"])
    ct = 0
    for years in range(max_years, 0, -1):
        for start_year in range(max_years - years + 1):
            for month in range(12):
                start_month = ref_month + month
                if start_month > 12:
                    start_month -= 12
                rp_start = pd.Timestamp(ref_year + start_year, start_month, 1)
                end = pd.Timestamp(ref_year + start_year + years, start_month, 1)
                possible = int((end - rp_start).total_seconds() / 600)
                ref_period_options[ct] = [rp_start, end, possible, years, np.nan, 0]
                ct += 1

    # pull primary met mast data
    ref_period_options = ref_period_options.T
    masts = pd.DataFrame(
        [i.MastData.df[i.MastData.primary_windSpeed_column] for i in masts], index=[i.id for i in masts]
    ).T

    # asses the reference period options
    sum_data_cols = []
    mae_cols = []
    for mast in masts:
        valid_points = []
        mean_ws = []
        for index, refPeriod in ref_period_options.iterrows():
            subset = masts[mast][(masts.index >= refPeriod["start"]) & (masts.index < refPeriod["end"])].dropna()
            valid_points.append(subset.size)
            mean_ws.append(subset.mean())
        ref_period_options[mast + " valid data"] = valid_points
        ref_period_options[mast + " recovery rate"] = (
            ref_period_options[mast + " valid data"] / ref_period_options["possible data points"]
        )
        ref_period_options[mast + " mean wind speed"] = mean_ws
        ref_period_options[mast + " MAE"] = ref_period_options[mast + " mean wind speed"] - masts[mast].mean()
        # save data point columns
        sum_data_cols.append(mast + " valid data")
        mae_cols.append(mast + " MAE")

    # complete nan columns
    ref_period_options["recovery rate"] = ref_period_options[sum_data_cols].sum(axis=1) / (
        ref_period_options["possible data points"] * len(masts.columns)
    )
    for data, mae in zip(sum_data_cols, mae_cols):
        ref_period_options["MAE"] += (
            ref_period_options[data] * ref_period_options[mae].replace(np.nan, 0)
        ) / ref_period_options["possible data points"]

    # select pareto points for recovery rate & mae fro each number of years
    ref_period_options["pareto"] = 0
    max_recovery = {i: [0, 0] for i in range(max_years, 0, -1)}
    for years in range(max_years, 0, -1):
        # for each number of years
        df_sub = ref_period_options[ref_period_options["length"] == years]
        for index, row in df_sub.iterrows():
            # find points on the pareto frontier
            if len(df_sub[(df_sub["MAE"] < row["MAE"]) & (df_sub["recovery rate"] > row["recovery rate"])]) == 0:
                ref_period_options["pareto"][index] = 1
                if row["recovery rate"] > max_recovery[years][1]:
                    max_recovery[years] = [index, row["recovery rate"]]
    recovery_min = max([value[1] for key, value in max_recovery.items()]) - 0.03
    max_recovery = {key: value for key, value in max_recovery.items() if value[1] >= recovery_min}
    default_period = max_recovery[max(max_recovery.keys())][0]
    return ref_period_options[ref_period_options["pareto"] == 1].drop(columns=["pareto"]), default_period


# noinspection PyProtectedMember
def cut_reference_period(masts: list, rp_start: pd.Timestamp, end: pd.Timestamp):
    """
    cut Mast.MastData.df object to the reference period

    Parameters
    ----------
    masts : list
        list of Mast objects.
    rp_start : pd.Timestamp
        start of data period.
    end : pd.Timestamp
        end of data period.

    Returns
    -------
    None.

    """
    for mast in masts:
        mast.MastData.df = mast.MastData.df[(mast.MastData.df.index >= rp_start) & (mast.MastData.df.index < end)]
        mast.MastData._get_weibull_fit()
        mast.MastData._get_directional_data()


def calculate_shear(masts: list, min_ws: float = 3, max_veer: float = 1000):
    """
    calculate directional and weighted directional (global) shear values for each
    Mast.MastData object

    Parameters
    ----------
    masts : list
        list of Mast objects.
    min_ws : float, optional
        minimum wind speed present in a time stamp used in determining
        the shear parameter. The default is 3.
    max_veer : float, optional
        maximum wind direction veer present in a time stamp used in determining
        the shear parameter. The default is 1000 (essentially no veer filter).

    Returns
    -------
    None.

    """
    # iterate through masts
    for mast in masts:
        columns = [value for key, value in mast.MastData.data_key.windSpeed_columns.items()] + ["directionSector"]
        sub_mast_data = mast.MastData.df[columns]
        # run veer filter
        if len(mast.MastData.data_key.windDirection_columns) > 1:
            max_height = max([key for key, value in mast.MastData.data_key.windDirection_columns.items()])
            min_height = min([key for key, value in mast.MastData.data_key.windDirection_columns.items()])
            sub_mast_data["veer"] = (
                (
                    mast.MastData.df[mast.MastData.data_key.windDirection_columns[max_height]]
                    - mast.MastData.df[mast.MastData.data_key.windDirection_columns[min_height]]
                )
                / (max_height - min_height)
                * 100
            )
            sub_mast_data["veer"][sub_mast_data["veer"] > max_veer] = np.nan
        # run wind speed filter
        for key, col in mast.MastData.data_key.windSpeed_columns.items():
            sub_mast_data[col][sub_mast_data[col] < min_ws] = np.nan
        # apply filters
        sub_mast_data.dropna(inplace=True)
        # initiate output dataframe
        df_shear = pd.DataFrame(index=[i for i in range(mast.MastData.directionSectors)])
        df_shear["valid time steps"] = [
            len(sub_mast_data[sub_mast_data["directionSector"] == i]) for i in range(mast.MastData.directionSectors)
        ]
        # iterate through sensor options
        for ht1, sensor1 in mast.MastData.data_key.windSpeed_columns.items():
            for ht2, sensor2 in mast.MastData.data_key.windSpeed_columns.items():
                if ht1 > ht2:
                    # don't repeat sensor combinations
                    sub_mast_data["shear"] = np.log(sub_mast_data[sensor1] / sub_mast_data[sensor2]) / np.log(ht1 / ht2)
                    df_shear[str(ht1) + " / " + str(ht2)] = [
                        sub_mast_data["shear"][sub_mast_data["directionSector"] == i].mean()
                        for i in range(mast.MastData.directionSectors)
                    ]
                    if (sensor1 == mast.MastData.primary_windSpeed_column) & (
                        sensor2 == mast.MastData.secondary_windSpeed_column
                    ):
                        mast.MastData.globalShear = (
                            df_shear[str(ht1) + " / " + str(ht2)] * df_shear["valid time steps"]
                        ).sum() / df_shear["valid time steps"].sum()
        mast.MastData.directionalShear = df_shear
        # TODO: add in displacement height


def por_plot(masts: list):
    """
    plot the period of record and missing data in each masts' primary wind speed sensor

    Parameters
    ----------
    masts : list
        list of Mast objects.

    Returns
    -------
    fig : str
        path to resulting data recovery figure.

    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    first_start = min([i.MastData.df.index[0] for i in masts])
    length = int((max([i.MastData.df.index[-1] for i in masts]) - first_start) / pd.Timedelta(minutes=10))
    my_df = pd.DataFrame(index=[first_start + pd.Timedelta(minutes=10 * i) for i in range(length + 1)])
    my_df["time int"] = [i for i in range(len(my_df))]
    plt.xlim([0, len(my_df)])
    plt.ylim([0, len(masts) / 5])
    plt.yticks([0.1 + 0.2 * i for i in range(len(masts))], [i.id for i in masts])
    x_ticks = my_df["time int"][
        ((my_df.index.month == 1) | (my_df.index.month == 7))
        & (my_df.index.day == 1)
        & (my_df.index.hour == 0)
        & (my_df.index.minute == 0)
    ]
    plt.xticks(x_ticks.values, x_ticks.index.strftime("%b %Y"), rotation=45)
    for ctr1, mast in enumerate(masts):
        my_df["tower"] = np.nan
        my_df["tower"] = mast.MastData.df[mast.MastData.primary_windSpeed_column]
        my_df["tower"][my_df.index < mast.MastData.df.index[0]] = 0
        my_df["tower"][my_df.index > mast.MastData.df.index[-1]] = 0
        my_df["tower"][min(mast.MastData.df.index[0] - pd.Timedelta(minutes=10), my_df.index[-1])] = np.nan
        my_df["tower"][min(mast.MastData.df.index[-1] + pd.Timedelta(minutes=10), my_df.index[-1])] = np.nan
        stops = my_df["time int"][my_df["tower"].isna()]
        last_start = 0
        for ind, val in stops.iteritems():
            if last_start > 0:
                ax.add_patch(
                    Rectangle((last_start + 1, 0.2 * ctr1), val - last_start - 1, 0.2, facecolor=my_colors[ctr1])
                )
            last_start = val
    return fig


if __name__ == "__main__":
    from BigFan import dataStructures as dS
    output_filepath = r"test\testWRA.xlsx"
    """ Initiate MET001 """
    mastID = "MET001"
    latitude = 25.0
    longitude = -120.0
    mastElevation = 1421
    input_file = r"test\test_windographerReader_custom.txt"
    wind_speed = {
        60: "Speed1",
        30: "Speed2",
    }
    wind_direction = {
        58: "Direction",
    }
    temperature = {
        2: "Temperature",
    }
    relative_humidity = {}
    pressure = {}
    dataKey = dS.DataKey(wind_speed, wind_direction, temperature, relative_humidity, pressure)
    MET001 = dS.Mast(mastID, latitude, longitude, mastElevation)
    MET001.add_data(input_file, dataKey)

    """ Initiate MET002 """
    mastID = "MET002"
    latitude = 45.1
    longitude = -120.1
    mastElevation = 1421
    input_file = r"test\test_windographerReader_custom.txt"
    wind_speed = {
        60: "Speed1",
        30: "Speed2",
    }
    wind_direction = {
        58: "Direction",
    }
    temperature = {
        2: "Temperature",
    }
    relative_humidity = {}
    pressure = {}
    dataKey = dS.DataKey(wind_speed, wind_direction, temperature, relative_humidity, pressure)
    MET002 = dS.Mast(mastID, latitude, longitude, mastElevation)
    MET002.add_data(input_file, dataKey)

    """ set masts and reference mast """
    my_masts = [MET001, MET002]
    reference_mast = MET001

    """ select whether gaps should be filled (True) or left (false) """
    gap_filling = False

    """ Set Power Curve for energy content comparisons """
    powerCurve = pd.read_csv(r"test\turbinePC.csv", index_col=0)

    """ Input Long-term data """
    sourceID = "vortex"
    latitude = 45.0
    longitude = -120.0
    input_file = r"test\test_vortexReader.txt"
    vortex = dS.Mast(sourceID, latitude, longitude, None)
    vortex.add_data(input_file, source="vortex")

    sourceID = "era5"
    latitude = 45.0
    longitude = -120.0
    input_file = r"test\test_windographerReader_ERA5.txt"
    era5 = dS.Mast(sourceID, latitude, longitude, None)
    era5.add_data(input_file, source="windog_download", time_shift=-6)

    sourceID = "merra2"
    latitude = 45.0
    longitude = -120.0
    input_file = r"test\test_windographerReader_MERRA2.txt"
    merra2 = dS.Mast(sourceID, latitude, longitude, None)
    merra2.add_data(input_file, source="windog_download", time_shift=-6)
    longTerm_sources = [vortex, era5, merra2]

    """ run WRA! """
    start = datetime.now()
    setUp_warnings = check_combined_data(my_masts, are_masts=True)
    # noinspection PyUnresolvedReferences
    setUp_warnings.append(check_combined_data(my_masts + longTerm_sources, are_masts=False))
    try:
        os.mkdir(os.path.join(os.path.abspath(os.path.dirname(__file__)), "figs"))
    except OSError:
        pass
    # TODO: check long-term data and delete what is only getting in the way
    long_term_correlations(my_masts, longTerm_sources, mcp.correlate_tls)
    # run concurrent period calc after all masts created
    df_concurrent, mcp_length, cp_figs = run_concurrent_period(my_masts, powerCurve, ref_mast=reference_mast.id)
    # run MCP after all masts created
    df_correlation = mcp.correlate_wind_speed_sensors(my_masts)
    if gap_filling:
        mcp.mcp_gap_fill(my_masts, df_correlation, mcp_method=mcp.correlate_tls, stop_length=mcp_length)
    # put in ref period selector
    refPeriod_options, default_selection = get_reference_periods(my_masts)
    for this_mast in my_masts:
        this_mast.MastData.refPeriod_start = refPeriod_options["start"][default_selection]
        this_mast.MastData.refPeriod_end = refPeriod_options["end"][default_selection]
    cut_reference_period(
        my_masts, refPeriod_options["start"][default_selection], refPeriod_options["end"][default_selection]
    )
    # TODO: user input here to select reference period or accept default (PLACE INTO 5a tab)
    # TODO: add in reference mast selection
    lt_ws = [i.MastData.df[i.MastData.primary_windSpeed_column].mean() for i in longTerm_sources]
    calculate_shear(my_masts)
    """ print warnings from code """
    for warning in setUp_warnings:
        print(warning)
    try:
        os.remove(os.path.join(os.path.abspath(os.path.dirname(__file__)), "figs"))
    except PermissionError:
        print("Could not remove figure folder")
    print("------------------")
    print()
    print("Code completed in: " + str(round((datetime.now() - start).total_seconds() / 60, 1)) + " minutes!")
    print()
    print("------------------")
