# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17, 2022

@author: Annalise Miller
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def plot_monthly_profile(plot_values: dict, concurrent: bool = False):
    """
    plot monthly wind speed averages of primary wind speed sensor

    Parameters
    ----------
    plot_values : dict
        dictionary of items to plot in diurnal form.
        key: mast object
        value: list of columns in key mast.MastData.df to plot
    concurrent : bool, optional
        if True, calculate monthly wind speed averages using only concurrent period data.
        The default is False.

    Returns
    -------
    fig : plt.plot
        path to the resulting monthly profile plot.

    """
    x = [i for i in range(1, 13)]
    fig = plt.figure(figsize=(10, 6.4))
    ct = 0
    for mast, columns in plot_values.items():
        for column in columns:
            if concurrent:
                if "CP" not in mast.MastData.df.columns:
                    raise KeyError("Please run concurrent period analysis before plotting concurrent data")
                y = [
                    mast.MastData.df[column][(mast.MastData.df.index.month == i) & (mast.MastData.df["CP"] == 1)].mean()
                    for i in range(1, 13)
                ]
            else:
                y = [mast.MastData.df[column][mast.MastData.df.index.month == i].mean() for i in range(1, 13)]
            plt.plot(x, y, color=my_colors[ct], label=mast.id + ": " + str(column))
            ct += 1
    plt.legend(loc="upper center", bbox_to_anchor=(0, -0.1, 1, 0), ncol=4)
    plt.title("Monthly Profile")
    plt.xlabel("Month")
    plt.ylabel("Value")
    plt.xlim([1, 12])
    plt.tight_layout()
    return fig


def profile_monthly(plot_values: dict, concurrent: bool = False):
    """
    create a table (pd.DataFrame) of the monthly wind speed averages of primary wind speed sensor

    Parameters
    ----------
    plot_values : dict
        dictionary of items to plot in diurnal form.
        key: mast object
        value: list of columns in key mast.MastData.df to plot
    concurrent : bool, optional
        if True, calculate monthly wind speed averages using only concurrent period data.
        The default is False.

    Returns
    -------
    fig : plt.plot
        path to the resulting monthly profile plot.

    """
    df_monthly = pd.DataFrame(index=[i for i in range(1, 13)])
    for mast, columns in plot_values.items():
        for column in columns:
            if concurrent:
                if "CP" not in mast.MastData.df.columns:
                    raise KeyError("Please run concurrent period analysis before plotting concurrent data")
                df_monthly[str(mast.id) + ": " + column] = [
                    mast.MastData.df[column][(mast.MastData.df.index.month == i) & (mast.MastData.df["CP"] == 1)].mean()
                    for i in range(1, 13)
                ]
            else:
                df_monthly[str(mast.id) + ": " + column] = [
                    mast.MastData.df[column][mast.MastData.df.index.month == i].mean() for i in range(1, 13)
                ]
    return df_monthly


def plot_diurnal_profile(plot_values: dict, concurrent: bool = False):
    """
    plot diurnal profile of selected data columns

    Parameters
    ----------
    plot_values : dict
        dictionary of items to plot in diurnal form.
        key: mast object
        value: list of columns in key mast.MastData.df to plot
    concurrent : bool, optional
        if True, calculate diurnal wind speed averages using only concurrent period data.
        The default is False.

    Returns
    -------
    fig : plt.plot
        diurnal distribution plot.

    """
    x = [i for i in range(24)]
    fig = plt.figure(figsize=(10, 6.4))
    ct = 0
    for mast, columns in plot_values.items():
        for column in columns:
            if concurrent:
                if "CP" not in mast.MastData.df.columns:
                    raise KeyError("Please run concurrent period analysis before plotting concurrent data")
                y = [
                    mast.MastData.df[column][(mast.MastData.df.index.hour == i) & (mast.MastData.df["CP"] == 1)].mean()
                    for i in range(24)
                ]
            else:
                y = [mast.MastData.df[column][mast.MastData.df.index.hour == i].mean() for i in range(24)]
            plt.plot(x, y, color=my_colors[ct], label=mast.id + ": " + str(column))
            ct += 1
    plt.legend(loc="upper center", bbox_to_anchor=(0, -0.1, 1, 0), ncol=4)
    plt.title("Diurnal Profile")
    plt.xlabel("Hour")
    plt.ylabel("Value")
    plt.xlim([0, 23])
    plt.tight_layout()
    return fig


def profile_diurnal(plot_values: dict, concurrent: bool = False):
    """
    create a table (pd.DataFrame) of the diurnal profile of selected data columns

    Parameters
    ----------
    plot_values : dict
        dictionary of items to plot in diurnal form.
        key: mast object
        value: list of columns in key mast.MastData.df to plot
    concurrent : bool, optional
        if True, calculate diurnal wind speed averages using only concurrent period data.
        The default is False.

    Returns
    -------
    df_diurnal : pd.DataFrame
        dataframe of the diurnal distribution of selected columns.
    """
    df_diurnal = pd.DataFrame(index=[i for i in range(24)])
    for mast, columns in plot_values.items():
        for column in columns:
            if concurrent:
                if "CP" not in mast.MastData.df.columns:
                    raise KeyError("Please run concurrent period analysis before plotting concurrent data")
                df_diurnal[str(mast.id) + ": " + column] = [
                    mast.MastData.df[column][(mast.MastData.df.index.hour == i) & (mast.MastData.df["CP"] == 1)].mean()
                    for i in range(24)
                ]
            else:
                df_diurnal[str(mast.id) + ": " + column] = [
                    mast.MastData.df[column][mast.MastData.df.index.hour == i].mean() for i in range(24)
                ]
    return df_diurnal


def profile_12x24(mast, column):
    """
    create a table (pd.DataFrame) of the 12x24 profile of a selected data column

    Parameters
    ----------
    mast : dataStructures.Mast
        the mast object you would like to create a 12x24 profile with
    column : str
        the name of the column withing mast.MastData.df that you would like
        to create a 12x24 profile of

    Returns
    -------
    df_12x24 : pd.DataFrame
        dataframe of the 12x24 distribution of the selected variable.
    """
    df_12x24 = pd.DataFrame(index=[i for i in range(24)], columns=[i for i in range(1, 13)])
    for month in range(1, 13):
        df_12x24[month] = [
            mast.MastData.df[column][
                (mast.MastData.df.index.month == month) & (mast.MastData.df.index.hour == hour)
            ].mean()
            for hour in range(24)
        ]
    return df_12x24


def plot_directional_profile(plot_values: dict, concurrent: bool = False):
    """
    plot directional profile of selected data columns

    Parameters
    ----------
    plot_values : dict
        dictionary of items to plot in diurnal form.
        key: mast object
        value: list of columns in key mast.MastData.df to plot
    concurrent : bool, optional
        if True, calculate diurnal wind speed averages using only concurrent period data.
        The default is False.

    Returns
    -------
    fig : plt.plot
        directional distribution plot.

    """
    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ct = 0
    for mast, columns in plot_values.items():
        x = [i * 2 * np.pi / mast.MastData.directionSectors for i in range(mast.MastData.directionSectors)]
        for column in columns:
            if concurrent:
                if "CP" not in mast.MastData.df.columns:
                    raise KeyError("Please run concurrent period analysis before plotting concurrent data")
                y = [
                    mast.MastData.df[column][
                        (mast.MastData.df["directionSector"] == i) & (mast.MastData.df["CP"] == 1)
                    ].mean()
                    for i in range(mast.MastData.directionSectors)
                ]
            else:
                y = [
                    mast.MastData.df[column][mast.MastData.df["directionSector"] == i].mean()
                    for i in range(mast.MastData.directionSectors)
                ]
            x.append(x[0])
            y.append(y[0])
            plt.plot(
                x,
                y,
                color=my_colors[ct],
                label=mast.id + ": " + str(mast.MastData.primary_windDirection_column_ht) + "m",
            )
            ct += 1
    plt.legend(loc="upper center", bbox_to_anchor=(0, -0.1, 1, 0), ncol=4)
    plt.title("Directional Distribution")
    plt.tight_layout()
    return fig


def profile_directional(plot_values: dict, concurrent: bool = False):
    """
    create a table (pd.DataFrame) of the directional profile of selected data columns

    Parameters
    ----------
    plot_values : dict
        dictionary of items to plot in diurnal form.
        key: mast object
        value: list of columns in key mast.MastData.df to plot
    concurrent : bool, optional
        if True, calculate diurnal wind speed averages using only concurrent period data.
        The default is False.

    Returns
    -------
    df_directional : pd.DataFrame
        dataframe of the directional statistics specified.

    """
    sectors = list(set([i.MastData.directionSectors for i in plot_values.keys]))
    if len(sectors) > 1:
        raise ValueError(
            "Please check mast.MastData.directionSectors. Ensure all masts have the same number of sectors specified."
        )
    sectors = [360 / (2 * sectors[0]) + (360 / sectors[0] * i) for i in range(sectors[0])]
    sectors = [sectors[-1]] + sectors
    df_directional = pd.DataFrame(index=[str(sectors[ct - 1]) + "-" + str(i) for ct, i in enumerate(sectors[1:])])
    for mast, columns in plot_values.items():
        for column in columns:
            if concurrent:
                if "CP" not in mast.MastData.df.columns:
                    raise KeyError("Please run concurrent period analysis before plotting concurrent data")
                df_directional[str(mast.id) + ": " + column] = [
                    mast.MastData.df[column][
                        (mast.MastData.df["directionSector"] == i) & (mast.MastData.df["CP"] == 1)
                    ].mean()
                    for i in range(mast.MastData.directionSectors)
                ]
            else:
                df_directional[str(mast.id) + ": " + column] = [
                    mast.MastData.df[column][mast.MastData.df["directionSector"] == i].mean()
                    for i in range(mast.MastData.directionSectors)
                ]
    return df_directional


def plot_hist_distribution(
    plot_values: dict,
    concurrent: bool = False,
    bin_width: float = 1.0,
    include_weibull: bool = False,
    fig_width: float = 10,
    fig_length: float = 6.4,
):
    """
    plot histogram distribution of selected sensor

    Parameters
    ----------
    plot_values : dict
        dictionary of items to plot in histogram form.
        key: mast object
        value: list of columns in key mast.MastData.df to plot in histogram
    concurrent : bool, optional
        if True, calculate wind speed distributions using only concurrent period data.
        The default is False.
    bin_width : float, optional
        width of bins for histogram.
        The defaults is 1 unit.
    include_weibull : bool, optional
        if True, include wind speed distribution from weibull fit in plot.
        The default is False.
    fig_width : float, optional
        width of output figure.
        The defaults is 10in.
    fig_length : float, optional
        length of output figure.
        The default is 6.4in.

    Returns
    -------
    fig : plt.plot
        histogram distribution plot.

    """
    fig = plt.figure(figsize=(fig_width, fig_length))
    ct = 0
    for mast, columns in plot_values.items():
        for column in columns:
            if concurrent:
                if "CP" not in mast.MastData.df.columns:
                    raise KeyError("Please run concurrent period analysis before plotting concurrent data")
                # get bin ends for plot
                min_val = mast.MastData.df[column][mast.MastData.df["CP"] == 1].min()
                low_end = int(min_val / bin_width)
                low_end -= bin_width / 2
                no_bins = int(
                    np.ceil((mast.MastData.df[column][mast.MastData.df["CP"] == 1].max() - low_end) / bin_width)
                )
                x = [low_end + ((i + 0.5) * bin_width) for i in range(no_bins)]
                y = [
                    len(
                        mast.MastData.df[
                            (mast.MastData.df[column] < low_end + ((i + 1) * bin_width))
                            & (mast.MastData.df[column] >= low_end + (i * bin_width))
                            & (mast.MastData.df["CP"] == 1)
                        ]
                    )
                    for i in range(no_bins + 1)
                ]
            else:
                # get bin ends for plot
                min_val = mast.MastData.df[column].min()
                low_end = int(min_val / bin_width)
                if low_end < 0:
                    low_end -= bin_width / 2
                else:
                    low_end -= bin_width / 2
                no_bins = int(np.ceil((mast.MastData.df[column].max() - low_end) / bin_width))
                x = [low_end + ((i + 0.5) * bin_width) for i in range(no_bins)]
                y = [
                    len(
                        mast.MastData.df[
                            (mast.MastData.df[column] < low_end + ((i + 1) * bin_width))
                            & (mast.MastData.df[column] >= low_end + (i * bin_width))
                        ]
                    )
                    for i in range(no_bins)
                ]
            # normalize values
            y = [i / sum(y) for i in y]
            plt.plot(x, y, color=my_colors[ct], label=mast.id + ": " + str(column))
            if include_weibull:
                flag = True
                wind_speed_columns = []
                for height, name in mast.MastData.data_key.windSpeedColumns.items():
                    wind_speed_columns.append(name)
                    if name == column:
                        flag = False
                if flag:
                    raise KeyError(
                        "Weibull plotting only available for wind speed columns: " + ", ".join(wind_speed_columns)
                    )
                y = [
                    (mast.MastData.k[column] / mast.MastData.A[column])
                    * pow(i / mast.MastData.A[column], mast.MastData.k[column] - 1)
                    * np.exp(-1 * pow(i / mast.MastData.A[column], mast.MastData.k[column]))
                    for i in x
                ]
                plt.plot(
                    x,
                    y,
                    color=my_colors[ct],
                    alpha=0.5,
                    linestyle="--",
                    label="Weibull fit for " + mast.id + ": " + str(mast.MastData.primary_windSpeed_column_ht) + "m",
                )
            ct += 1
    plt.legend(loc="upper center", bbox_to_anchor=(0, -0.1, 1, 0), ncol=4)
    plt.title("PDF")
    plt.ylabel("Frequency (%)")
    plt.tight_layout()
    return fig


def hist_distribution(
    plot_values: dict, concurrent: bool = False, bin_width: float = 1.0, include_weibull: bool = False,
):
    """
    plot histogram distribution of selected sensor

    Parameters
    ----------
    plot_values : dict
        dictionary of items to plot in histogram form.
        key: mast object
        value: list of columns in key mast.MastData.df to plot in histogram
    concurrent : bool, optional
        if True, calculate wind speed distributions using only concurrent period data.
        The default is False.
    bin_width : float, optional
        width of bins for histogram.
        The defaults is 1 unit.
    include_weibull : bool, optional
        if True, include wind speed distribution from weibull fit in plot.
        The default is False.

    Returns
    -------
    df : pd.DataFrame
        histogram data in table (pd.DataFrame) form

    """
    min_val = 100
    max_val = -100
    # determine range of data
    for mast, columns in plot_values.items():
        for column in columns:
            if concurrent:
                if mast.MastData.df[column][mast.MastData.df["CP"] == 1].min() < min_val:
                    min_val = mast.MastData.df[column][mast.MastData.df["CP"] == 1].min()
                if mast.MastData.df[column][mast.MastData.df["CP"] == 1].max() < max_val:
                    max_val = mast.MastData.df[column][mast.MastData.df["CP"] == 1].max()
            else:
                if mast.MastData.df[column].min() < min_val:
                    min_val = mast.MastData.df[column].min()
                if mast.MastData.df[column].max() < max_val:
                    max_val = mast.MastData.df[column].max()

    low_end = int(min_val / bin_width)
    low_end -= bin_width / 2
    no_bins = int(np.ceil((max_val - low_end) / bin_width))
    df_hist = pd.DataFrame(index=[low_end + ((i + 0.5) * bin_width) for i in range(no_bins)])
    for mast, columns in plot_values.items():
        for column in columns:
            if concurrent:
                if "CP" not in mast.MastData.df.columns:
                    raise KeyError("Please run concurrent period analysis before plotting concurrent data")
                # get bin ends for plot
                df_hist[str(mast.id) + ": " + column] = [
                    len(
                        mast.MastData.df[
                            (mast.MastData.df[column] < low_end + ((i + 1) * bin_width))
                            & (mast.MastData.df[column] >= low_end + (i * bin_width))
                            & (mast.MastData.df["CP"] == 1)
                        ]
                    )
                    for i in range(no_bins + 1)
                ]
            else:
                df_hist[str(mast.id) + ": " + column] = [
                    len(
                        mast.MastData.df[
                            (mast.MastData.df[column] < low_end + ((i + 1) * bin_width))
                            & (mast.MastData.df[column] >= low_end + (i * bin_width))
                        ]
                    )
                    for i in range(no_bins)
                ]
            # normalize values
            df_hist[str(mast.id) + ": " + column] /= df_hist[str(mast.id) + column].sum()
            if include_weibull:
                flag = True
                wind_speed_columns = []
                for height, name in mast.MastData.data_key.windSpeedColumns.items():
                    wind_speed_columns.append(name)
                    if name == column:
                        flag = False
                if flag:
                    raise KeyError(
                        "Weibull plotting only available for wind speed columns: " + ", ".join(wind_speed_columns)
                    )
                df_hist[str(mast.id) + ": " + column + ", weibull fit"] = [
                    (mast.MastData.k[column] / mast.MastData.A[column])
                    * pow(i / mast.MastData.A[column], mast.MastData.k[column] - 1)
                    * np.exp(-1 * pow(i / mast.MastData.A[column], mast.MastData.k[column]))
                    for i in df_hist.index
                ]
    return df_hist


def plot_wind_rose(masts: list, concurrent: bool = False):
    """
    plot wind direction rose of primary wind direction sensor

    Parameters
    ----------
    masts : list
        list of Mast objects.
    concurrent : bool, optional
        if True, calculate wind direction rose using only concurrent period data.
        The default is False.

    Returns
    -------
    figPath : str
        path to the resulting wind direction rose plot.

    """
    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    for ct, mast in enumerate(masts):
        x = [i * 2 * np.pi / mast.MastData.directionSectors for i in range(mast.MastData.directionSectors)]
        if concurrent:
            y = [
                len(mast.MastData.df[(mast.MastData.df["directionSector"] == i) & (mast.MastData.df["CP"] == 1)])
                for i in range(mast.MastData.directionSectors)
            ]
        else:
            y = [
                len(mast.MastData.df[mast.MastData.df["directionSector"] == i])
                for i in range(mast.MastData.directionSectors)
            ]
        y = [i / sum(y) for i in y]
        x.append(x[0])
        y.append(y[0])
        plt.plot(
            x, y, color=my_colors[ct], label=mast.id + ": " + str(mast.MastData.primary_windDirection_column_ht) + "m"
        )
    plt.legend(loc="upper center", bbox_to_anchor=(0, -0.1, 1, 0), ncol=4)
    plt.title("Wind Rose")
    plt.tight_layout()
    return fig


def create_wind_rose(masts: list, concurrent: bool = False):
    """
    create a table (pd.DataFrame) of the wind direction rose from the primary wind direction sensor

    Parameters
    ----------
    masts : list
        list of Mast objects.
    concurrent : bool, optional
        if True, calculate wind direction rose using only concurrent period data.
        The default is False.

    Returns
    -------
    df_wind_rose : pd.DataFrame
        dataframe of wind rose data

    """
    sectors = list(set([i.MastData.directionSectors for i in masts]))
    if len(sectors) > 1:
        raise ValueError(
            "Please check mast.MastData.directionSectors. Ensure all masts have the same number of sectors specified."
        )
    sectors = [360 / (2 * sectors[0]) + (360 / sectors[0] * i) for i in range(sectors[0])]
    sectors = [sectors[-1]] + sectors
    df_wind_rose = pd.DataFrame(index=[str(sectors[ct - 1]) + "-" + str(i) for ct, i in enumerate(sectors[1:])])
    for ct, mast in enumerate(masts):
        if concurrent:
            df_wind_rose[str(mast.id)] = [
                len(mast.MastData.df[(mast.MastData.df["directionSector"] == i) & (mast.MastData.df["CP"] == 1)])
                for i in range(mast.MastData.directionSectors)
            ]
        else:
            df_wind_rose[str(mast.id)] = [
                len(mast.MastData.df[mast.MastData.df["directionSector"] == i])
                for i in range(mast.MastData.directionSectors)
            ]
    return df_wind_rose


if __name__ == "__main__":
    from BigFan import dataStructures as dS

    sourceID = "era5"
    latitude = 41.237822
    longitude = -89.923782
    input_file = r"..\tests\era5Test_IL.csv"
    wind_speed = {
        100: "WS_100m",
    }
    wind_direction = {
        100: "WD_100m",
    }
    temperature = {
        2: "t2m",
    }
    relative_humidity = {}
    pressure = {0: "sp"}
    data_key = dS.DataKey(wind_speed, wind_direction, temperature, relative_humidity, pressure)
    era5 = dS.Mast(sourceID, latitude, longitude, None)
    era5.add_data(input_file, data_key, source="csv", time_shift=-8)
    sourceID = "merra2"
    latitude = 41.237822
    longitude = -89.923782
    input_file = r"..\tests\merra2Test_IL.csv"
    wind_speed = {
        50: "WS50M",
    }
    wind_direction = {
        50: "WD50M",
    }
    temperature = {
        2: "T2M",
    }
    relative_humidity = {0: "RH2M"}
    pressure = {0: "PS"}
    data_key = dS.DataKey(wind_speed, wind_direction, temperature, relative_humidity, pressure)
    merra2 = dS.Mast(sourceID, latitude, longitude, None)
    merra2.add_data(input_file, data_key, source="csv")
    # actual testing
    plot_vals = {merra2: ["WS50M"], era5: ["WS_100m"]}
    # plot_hist_distribution(plot_values=plot_vals, include_weibull=True)
    # plot_diurnal_profile(plot_vals)
    # plot_monthly_profile(plot_vals)
    # plot_directional_profile(plot_vals)
    # plot_wind_rose([merra2, era5])
    ws_12x24 = profile_12x24(merra2, "WS50M")
