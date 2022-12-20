# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17, 2022

@author: Annalise Miller
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from BigFan import windResourceAnalysis as wra  # only for testing

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


def plot_monthly_ws(masts: list, concurrent: bool = False):
    """
    plot monthly wind speed averages of primary wind speed sensor

    Parameters
    ----------
    masts : list
        list of Mast objects.
    concurrent : bool, optional
        if True, calculate monthly wind speed averages using only concurrent period data.
        The default is False.

    Returns
    -------
    figPath : str
        path to the resulting monthly wind speed plot.

    """
    x = [i for i in range(1, 13)]
    plt.figure(figsize=(10, 6.4))
    for ct, mast in enumerate(masts):
        if concurrent:
            y = [
                mast.MastData.df[mast.MastData.primary_windSpeed_column][
                    (mast.MastData.df.index.month == i) & (mast.MastData.df["CP"] == 1)
                ].mean()
                for i in range(1, 13)
            ]
        else:
            y = [
                mast.MastData.df[mast.MastData.primary_windSpeed_column][mast.MastData.df.index.month == i].mean()
                for i in range(1, 13)
            ]
        plt.plot(x, y, color=my_colors[ct], label=mast.id + ": " + str(mast.MastData.primary_windSpeed_column_ht) + "m")
    plt.legend(loc="upper center", bbox_to_anchor=(0, -0.1, 1, 0), ncol=4)
    plt.title("Monthly mean wind speed")
    plt.xlabel("Month")
    plt.ylabel("Mean wind speed (m/s)")
    plt.xlim([1, 12])
    ct = 0
    while os.path.exists(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "figs", "MonthlyWS" + str(ct) + ".png")
    ):
        ct += 1
    fig_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "figs", "MonthlyWS" + str(ct) + ".png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def plot_diurnal_ws(masts: list, concurrent: bool = False):
    """
    plot diurnal wind speed averages of primary wind speed sensor

    Parameters
    ----------
    masts : list
        list of Mast objects.
    concurrent : bool, optional
        if True, calculate diurnal wind speed averages using only concurrent period data.
        The default is False.

    Returns
    -------
    figPath : str
        path to the resulting diurnal wind speed plot.

    """
    x = [i for i in range(24)]
    plt.figure(figsize=(10, 6.4))
    for ct, mast in enumerate(masts):
        if concurrent:
            y = [
                mast.MastData.df[mast.MastData.primary_windSpeed_column][
                    (mast.MastData.df.index.hour == i) & (mast.MastData.df["CP"] == 1)
                ].mean()
                for i in range(24)
            ]
        else:
            y = [
                mast.MastData.df[mast.MastData.primary_windSpeed_column][mast.MastData.df.index.hour == i].mean()
                for i in range(24)
            ]
        plt.plot(x, y, color=my_colors[ct], label=mast.id + ": " + str(mast.MastData.primary_windSpeed_column_ht) + "m")
    plt.legend(loc="upper center", bbox_to_anchor=(0, -0.1, 1, 0), ncol=4)
    plt.title("Diurnal mean wind speed")
    plt.xlabel("Hour")
    plt.ylabel("Mean wind speed (m/s)")
    plt.xlim([0, 23])
    ct = 0
    while os.path.exists(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "figs", "DiurnalWS" + str(ct) + ".png")
    ):
        ct += 1
    fig_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "figs", "DiurnalWS" + str(ct) + ".png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def plot_hist_distribution(
    plot_values: dict,
    concurrent: bool = False,
    include_weibull: bool = False,
    fig_width: float = 10,
    fig_length: float = 6.4,
    bin_width: float = 1.0,
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
    include_weibull : bool, optional
        if True, include wind speed distribution from weibull fit in plot.
        The default is False.
    fig_width : float, optional
        width of output figure.
        The defaults is 10in.
    fig_length : float, optional
        length of output figure.
        The default is 6.4in.
    bin_width : float, optional
        width of bins for histogram.
        The defaults is 1 unit.

    Returns
    -------
    fig : str
        histogram distribution plot.

    """
    fig = plt.figure(figsize=(fig_width, fig_length))
    ct = 0
    for mast, columns in enumerate(plot_values):
        for column in columns:
            if concurrent:
                # get bin ends for plot
                min_val = mast.MastData.df[column][mast.MastData.df["CP"] == 1].min()
                low_end = int(min_val / bin_width)
                if low_end < 0:
                    low_end -= bin_width / 2
                else:
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
                    for i in range(no_bins + 1)
                ]
            # normalize values
            y = [i / sum(y) for i in y]
            plt.plot(x, y, color=my_colors[ct], label=mast.id + ": " + str(column))
            ct += 1
            if include_weibull:
                x = [0] + x
                y = [0] + [
                    (mast.k / mast.a) * pow(i / mast.a, mast.k - 1) * np.exp(-1 * pow(i / mast.a, mast.k)) for i in x
                ]
                plt.plot(
                    x,
                    y,
                    color=my_colors[ct],
                    alpha=0.5,
                    linestyle="--",
                    label="Weibull fit for " + mast.id + ": " + str(mast.MastData.primary_windSpeed_column_ht) + "m",
                )
    plt.legend(loc="upper center", bbox_to_anchor=(0, -0.1, 1, 0), ncol=4)
    return fig


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
    ct = 0
    while os.path.exists(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "figs", "windRose" + str(ct) + ".png")
    ):
        ct += 1
    fig_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "figs", "windRose" + str(ct) + ".png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def basic_quad_plots(masts: list, concurrent=False):
    """
    plot monthly wind speeds, diurnal wind speeds, wind speed distributions,
    and wind direction rose of primary wind speed and wind direction sensors

    Parameters
    ----------
    masts : list
        list of Mast objects.
    concurrent : bool, optional
        if True, create plots using only concurrent period data.
        The default is False.

    Returns
    -------
    figs : list
        list of paths to resulting figures.

    """
    figs = [
        plot_monthly_ws(masts, concurrent),
        plot_diurnal_ws(masts, concurrent),
        plot_ws_distribution(masts, concurrent),
        plot_wind_rose(masts, concurrent),
    ]
    return figs


if __name__ == "__main__":
    sourceID = "era5"
    latitude = 45.0
    longitude = -120.0
    input_file = r"C:\Users\Annalise\Downloads\ERA5_test.csv"
    era5 = wra.Mast(sourceID, latitude, longitude, None)
    era5.add_data(input_file, source="csv", time_shift=-8)
    # actual testing
    plot_hist_distribution(plot_values={era5: ["WS_100"]})
