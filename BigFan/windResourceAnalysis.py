# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15, 2022

@author: Annalise Miller
"""
import dataclasses
from typing import Dict, Optional, Union, Callable
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm, f
from scipy.special import gamma
from scipy.interpolate import griddata
from utm import from_latlon
import geomag
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from datetime import datetime


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


class DataKey:
    def __init__(self):
        windSpeed_columns: Dict[Union[int, float], str]
        windDirection_columns: Dict[Union[int, float], str]
        temperature_columns: Dict[Union[int, float], str]
        relativeHumidity_columns: Dict[Union[int, float], str]
        pressure_columns: Dict[Union[int, float], str]

    def __post_init__(self):
        # set heights of defaults
        # info that will definitely be on a mast
        object.__setattr__(
            self, "default_primary_windSpeed_column_ht", self.get_default_primary_key(self.windSpeed_columns)
        )
        object.__setattr__(
            self, "default_secondary_windSpeed_column_ht", self.get_default_secondary_key(self.windSpeed_columns)
        )
        object.__setattr__(
            self, "default_primary_windDirection_column_ht", self.get_default_primary_key(self.windDirection_columns)
        )
        object.__setattr__(
            self, "default_primary_temperature_column_ht", self.get_default_primary_key(self.temperature_columns)
        )

        # info that may not be on a mast
        object.__setattr__(
            self,
            "default_primary_relativeHumidity_column_ht",
            self.get_default_primary_key(self.relativeHumidity_columns),
        )
        object.__setattr__(
            self, "default_primary_pressure_column_ht", self.get_default_primary_key(self.pressure_columns)
        )

        # set column names of defaults
        # info that will definitely be on a mast
        object.__setattr__(
            self, "default_primary_windSpeed_column", self.windSpeed_columns[self.default_primary_windSpeed_column_ht]
        )
        object.__setattr__(
            self,
            "default_secondary_windSpeed_column",
            self.windSpeed_columns[self.default_secondary_windSpeed_column_ht],
        )
        object.__setattr__(
            self,
            "default_primary_windDirection_column",
            self.windDirection_columns[self.default_primary_windDirection_column_ht],
        )
        object.__setattr__(
            self,
            "default_primary_temperature_column",
            self.temperature_columns[self.default_primary_temperature_column_ht],
        )

        # info that may not be on a mast
        if len(self.relativeHumidity_columns) > 0:
            object.__setattr__(
                self,
                "default_primary_relativeHumidity_column",
                self.relativeHumidity_columns[self.default_primary_relativeHumidity_column_ht],
            )
        else:
            object.__setattr__(self, "default_primary_relativeHumidity_column", None)
        if len(self.pressure_columns) > 0:
            object.__setattr__(
                self, "default_primary_pressure_column", self.pressure_columns[self.default_primary_pressure_column_ht]
            )
        else:
            object.__setattr__(self, "default_primary_pressure_column", None)

    @staticmethod
    def get_default_primary_key(possible_keys: Dict[Union[int, float], str]) -> Optional[str]:
        if len(possible_keys) > 0:
            # return the name of the topmost sensor of a given type
            return max(possible_keys.keys())
        else:
            return None

    @staticmethod
    def get_default_secondary_key(possible_keys: Dict[Union[int, float], str]) -> Optional[str]:
        # determine the default lower wind speed sensor for shear
        target = max(possible_keys.keys()) - 20
        difference = 1e6
        secondary = None
        for key, value in possible_keys.items():
            if abs(key - target) < difference:
                secondary = key
        return secondary


class MastData:
    def __init__(self, file_path: str, data_key: Optional[DataKey] = None, source: str = "windog", time_shift: int = 0):
        self.file_path: str = file_path
        self.directionSectors = 16
        if source == "windog":
            self.df: Optional[pd.DataFrame] = self._add_timeSeries(file_path)
            if data_key is None:
                raise ValueError(
                    "Windographer files that do not originate from the data downloader function must include a data key!"
                )
            # decrypt data
            self.data_key: DataKey = data_key
        elif source == "vortex":
            self.df: Optional[pd.DataFrame] = self._add_vortex(file_path)
            data_key = DataKey({60: "M(m/s)"}, {60: "D(deg)"}, {60: "T(C)"}, {60: "PRE(hPa)"}, {60: "RH(%)"},)
            self.data_key: DataKey = data_key
        elif source == "windog_download":
            self.df: Optional[pd.DataFrame] = self._add_timeSeries(file_path, "\t")
            if "era5" in file_path.lower():
                data_key = DataKey(
                    {100: "Speed_100m [m/s]"},
                    {100: "Direction_100m [degrees]"},
                    {2: "Temperature_2m [degrees C]"},
                    {0: "Pressure_0m [kPa]"},
                    {},
                )
            elif "merra2" in file_path.lower():
                data_key = DataKey(
                    {60: "Speed_50m [m/s]"},
                    {60: "Direction_50m [degrees]"},
                    {10: "Temperature_10m [degrees C]", 2: "Temperature_2m [degrees C]"},
                    {0: "Pressure_0m [kPa]"},
                    {},
                )
            else:
                raise ValueError(
                    'Please ensure that "merra2" or "era5" (case insensitive) is in the file name of the data from the windographer data downloader.'
                )
            self.data_key: DataKey = data_key
        elif source == "csv":
            self.df: Optional[pd.DataFrame] = self._add_csv(file_path)
            if data_key is None:
                raise ValueError("csv input must include a data key!")
            # decrypt data
            self.data_key: DataKey = data_key
        else:
            raise ValueError(
                "source input: "
                + str(source)
                + ' is not recognized. Please use "windog", "vortex" or "windog_download", or "csv"'
            )

        # meta data on inputs
        self.data_start: pd.Timestamp = self.df.index[0]
        self.data_end: pd.Timestamp = self.df.index[-1]
        self.refPeriod_start: pd.Timestamp = self.df.index[0]
        self.refPeriod_end: pd.Timestamp = self.df.index[-1]

        # validate data
        self._validate_dataKey(self.df.columns.tolist(), data_key)
        # set relevant column names
        self.primary_windSpeed_column: str = data_key.default_primary_windSpeed_column
        self.secondary_windSpeed_column: str = data_key.default_secondary_windSpeed_column
        self.primary_windDirection_column: str = data_key.default_primary_windDirection_column
        self.primary_temperature_column: str = data_key.default_primary_temperature_column
        self.primary_relativeHumidity_column: str = data_key.default_primary_relativeHumidity_column
        self.primary_pressure_column: str = data_key.default_primary_pressure_column
        # set relative column heights
        self.primary_windSpeed_column_ht: Union[int, float] = data_key.default_primary_windSpeed_column_ht
        self.secondary_windSpeed_column_ht: Union[int, float] = data_key.default_secondary_windSpeed_column_ht
        self.primary_windDirection_column_ht: Union[int, float] = data_key.default_primary_windDirection_column_ht
        self.primary_temperature_column_ht: Union[int, float] = data_key.default_primary_temperature_column_ht
        self.primary_relativeHumidity_column_ht: Union[int, float] = data_key.default_primary_relativeHumidity_column_ht
        self.primary_pressure_column_ht: Union[int, float] = data_key.default_primary_pressure_column_ht

        # key data analysis outputs
        self.globalShear: Optional[float] = None
        self.directionalShear: Optional[float] = None
        # TODO: select these and fill in variables
        self.longTerm_measurementHeight_windSpeed: Optional[float] = None
        self.longTerm_hubHeight_windSpeed: Optional[float] = None

        # add direction sectors
        self._get_directionSector()
        self.directional_WindSpeed = self._get_directionalData()

        # get weibull distribution factors
        self._get_weibullFit()

        # apply time shift
        self.time_shift = time_shift
        if time_shift != 0:
            self.df.index = self.df.index + pd.Timedelta(hours=time_shift)

        # set up for mast relationships
        self.longTerm_corr = pd.DataFrame(index=["Time steps", "Offset", "Slope", "R2"])
        self.mast_corr = pd.DataFrame(index=["Time steps", "Offset", "Slope", "R2"])

    @staticmethod
    def _add_timeSeries(file_path: str, seperator: str = ","):
        # read windog txt file format
        with open(file_path, newline="") as my_file:
            all_lines = my_file.readlines()
        for ct, line in enumerate(all_lines):
            if all_lines[ct][:9].lower() == "date/time":
                save_start = ct + 1
                cols = all_lines[ct].encode("utf-8", errors="ignore").decode()
                cols = cols.strip("\r\n").split(seperator)
                break
        try:
            df = pd.read_csv(
                file_path,
                skiprows=save_start,
                names=cols,
                sep=seperator,
                index_col=0,
                parse_dates=True,
                infer_datetime_format=True,
                encoding_errors="ignore",
            )
        except TypeError:
            df = pd.read_csv(
                file_path,
                skiprows=save_start,
                names=cols,
                sep=seperator,
                index_col=0,
                parse_dates=True,
                infer_datetime_format=True,
            )
        df = df[[k for k in df.keys() if not all(pd.isna(df[k]))]]
        return df.astype(float)

    @staticmethod
    def _add_vortex(file_path: str):
        # read vortex file
        with open(file_path, newline="") as my_file:
            all_lines = my_file.readlines()
        flag = False
        df = []
        for ct, line in enumerate(all_lines):
            if flag:
                line = line.encode("utf-8", errors="ignore").decode()
                line = line.strip("\r\n").split(" ")
                while "" in line:
                    line.remove("")
                df.append(line)
            if all_lines[ct][:8] == "YYYYMMDD":
                flag = True
                cols = line.encode("utf-8", errors="ignore").decode()
                cols = cols.strip("\r\n").split(" ")
                while "" in cols:
                    cols.remove("")
        df = pd.DataFrame(df, columns=cols)
        df = df[[k for k in df.keys() if not all(pd.isna(df[k]))]]
        start_time = pd.to_datetime(df["YYYYMMDD"][0] + df["HHMM"][0], format="%Y%m%d%H%M")
        df.index = [start_time + pd.Timedelta(hours=i) for i in range(len(df))]
        df.drop(columns=["YYYYMMDD", "HHMM"], inplace=True)
        return df.astype(float)

    @staticmethod
    def _add_csv(file_path: str):
        # read csv file
        try:
            df = pd.read_csv(
                file_path, index_col=0, parse_dates=True, infer_datetime_format=True, encoding_errors="ignore"
            )
        except TypeError:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True, infer_datetime_format=True)
        df = df[[k.strip() for k in df.keys() if not all(pd.isna(df[k]))]]
        return df.astype(float)

    @staticmethod
    def _validate_dataKey(columns: list, data_key: DataKey):
        sensor_types = list(data_key.windSpeed_columns.values())
        sensor_types.extend(list(data_key.windDirection_columns.values()))
        sensor_types.extend(list(data_key.temperature_columns.values()))
        sensor_types.extend(list(data_key.relativeHumidity_columns.values()))
        sensor_types.extend(list(data_key.pressure_columns.values()))
        sensor_types = set(sensor_types)
        sensor_types = [i for i in sensor_types if i not in columns]
        if len(sensor_types) > 0:
            raise ValueError(
                "The following input columns were not found in the windographer file: " + ", ".join(sensor_types)
            )

    def _get_directionSector(self):
        bin_width = 360.0 / self.directionSectors
        self.df["directionSector"] = (self.df[self.primary_windDirection_column] + (bin_width / 2)) / bin_width
        self.df["directionSector"] = self.df["directionSector"].fillna(999)
        self.df["directionSector"] = self.df["directionSector"].astype(int)
        self.df["directionSector"][self.df["directionSector"] == self.directionSectors] = 0
        self.df["directionSector"] = self.df["directionSector"].replace(999, np.nan)

    def _get_weibullFit(self, threshold: float = 1e-5):
        mast_k = {}
        mast_A = {}
        for key, sensor in self.data_key.windSpeed_columns.items():
            mean_val = self.df[sensor].mean()
            total_vals = len(self.df[sensor].dropna())
            x = len(self.df[self.df[sensor] > mean_val]) / total_vals
            k = 2.0
            step = 0.1
            flag = True
            ct = 0
            max_ct = 50
            move = [0, 0]
            while (flag) & (ct < max_ct):
                A = pow((pow(self.df[sensor], 3).sum() / total_vals) / gamma(3 / k + 1), 1.0 / 3)
                numerator = pow((self.df[sensor].sum() / total_vals) / A, k)
                test_val = np.log(x) + numerator
                if abs(test_val) < threshold:
                    flag = False
                    mast_k[sensor] = k
                    mast_A[sensor] = A
                elif test_val < 0:
                    k -= step
                    move = [move[-1], 1]
                else:
                    k += step
                    move = [move[-1], -1]
                if sum(move) == 0:
                    step /= 2
                ct += 1
        self.k = mast_k
        self.A = mast_A
        pass

    def _get_directionalData(self) -> pd.DataFrame:
        self.directional_WindSpeed = pd.DataFrame(index=[i for i in range(self.directionSectors)])
        self.directional_WindSpeed["wind speed (m/s)"] = [
            self.df[self.primary_windSpeed_column][self.df[self.primary_windDirection_column] == i].mean()
            for i in range(self.directionSectors)
        ]
        self.directional_WindSpeed["frequency"] = [
            len(self.df[self.df[self.primary_windDirection_column] == i].dropna()) for i in range(self.directionSectors)
        ]
        self.directional_WindSpeed["frequency"] = (
            self.directional_WindSpeed["frequency"] / self.directional_WindSpeed["frequency"].sum()
        )
        return self.directional_WindSpeed

    def calc_monthlyWindSpeed(self) -> pd.DataFrame:
        # initiate data frame
        monthly_df = self.df[self.data_key.windSpeed_columns.values()].resample("M").agg(["mean", "count"])
        possible_dp = monthly_df.index.day.values * 24 * 6
        for sensor in self.data_key.windSpeed_columns.values():
            # calculate data coverage rate
            monthly_df[(sensor, "DCR")] = monthly_df[sensor]["count"] / possible_dp
        # remove time stamps from incomplete months at beginning and end
        possible_dp[0] -= (
            self.df.index[0] - pd.Timestamp(self.df.index[0].year, self.df.index[0].month, 1)
        ).total_seconds() / 600
        if self.df.index[-1].month == 12:
            possible_dp[-1] -= (
                pd.Timestamp(self.df.index[-1].year + 1, 1, 1) - self.df.index[-1]
            ).total_seconds() / 600
        else:
            possible_dp[-1] -= (
                (
                    pd.Timestamp(self.df.index[-1].year, self.df.index[-1].month + 1, 1) - self.df.index[-1]
                ).total_seconds()
                / 600
            ) - 1
        monthly_df["possible_dataPoints"] = possible_dp
        order = [("possible_dataPoints", "")]
        for sensor in self.data_key.windSpeed_columns.values():
            # calculate data availability
            monthly_df[(sensor, "DRR")] = monthly_df[sensor]["count"] / monthly_df["possible_dataPoints"]
            order.extend([(sensor, "count"), (sensor, "DCR"), (sensor, "DRR"), (sensor, "mean")])
        return monthly_df[order]

    @staticmethod
    def check_linearRegression_trend(series: pd.Series, threshold: float = 0.9) -> bool:
        year = np.array(series.index.year)
        sxx = pow(year - year.mean(), 2).sum()
        syy = pow(series - series.mean(), 2).sum()
        sxy = ((series - series.mean()) * (year - year.mean())).sum()
        if series.size == 2:
            sig_y = 0
        elif series.size > 2:
            sig_y = pow((syy - (pow(sxy, 2) / sxx)) / (series.size - 2), 0.5)
        else:
            raise ValueError("Fewer than 2 years found in time series data")
        m = sxy / sxx
        k = abs(m) / (sig_y / pow(sxx, 0.5))
        dist = 2 * norm.cdf(k) - 1
        return abs(dist) > threshold

    @staticmethod
    def check_mannKendall_trend(series, threshold: float = 0.9) -> bool:
        series = pd.Series([series[ct] - series[ct - 1] for ct in range(1, len(series))])
        group_add = sum([i * (i - 1) * (2 * i + 5) for i in series.value_counts()])
        total_vals = len(series)
        series[series > 0] = 1
        series[series < 0] = -1
        s = sum([series[ct + 1 :].sum() for ct in range(len(series) - 1)])
        if s > 0:
            s -= 1
        elif s < 0:
            s += 1
        sig_s = pow((total_vals * (total_vals - 1) * (2 * total_vals + 5) - group_add) / 18, 0.5)
        dist = 2 * abs(norm.cdf(s / sig_s)) - 1
        return abs(dist) > threshold

    @staticmethod
    def check_easterlingPeterson_discontinuity(series, threshold: float = 0.9) -> bool:
        if len(series) < 7:
            raise ValueError("Insufficient data in long-term series (<7 years)")
        series.index = [i for i in range(series.size)]
        linear_regressor = LinearRegression()
        reg0 = linear_regressor.fit(series.index.values.reshape(-1, 1), series.values.reshape(-1, 1))
        rss0 = pow(series - series.index.values * reg0.coef_[0][0] + reg0.intercept_[0], 2).sum()
        dof = len(series) - 4
        max_rss = -1
        for break_pt in range(3, len(series) - 3):
            rss1 = 0
            linear_regressor = LinearRegression()
            reg1 = linear_regressor.fit(
                series.index.values[:break_pt].reshape(-1, 1), series.values[:break_pt].reshape(-1, 1)
            )
            rss1 += pow(
                series[:break_pt] - series.index.values[:break_pt] * reg1.coef_[0][0] + reg1.intercept_[0], 2
            ).sum()
            linear_regressor = LinearRegression()
            reg2 = linear_regressor.fit(
                series.index.values[break_pt:].reshape(-1, 1), series.values[break_pt:].reshape(-1, 1)
            )
            rss1 += pow(
                series[break_pt:] - series.index.values[break_pt:] * reg2.coef_[0][0] + reg2.intercept_[0], 2
            ).sum()
            if rss1 > max_rss:
                max_rss = rss1 * 1
                u = ((rss0 - rss1) / 3) / (rss1 / dof)
        return u > f.cdf(threshold, 3, dof)

    def verify_longTerm_viability(self, series_input: pd.DataFrame, threshold: float = 0.95) -> (bool, bool, bool):
        # set up annual data
        max_vals = pd.Timedelta(days=365) / (series_input.index[1] - series_input.index[0])
        df = pd.DataFrame([])
        df["WS"] = series_input.resample("A").mean()
        df["drr"] = series_input.resample("A").count() / max_vals
        df = df["WS"][df["drr"] > threshold]
        # run tests
        linReg = self.check_linearRegression_trend(df)
        mannKendall = self.check_mannKendall_trend(df)
        eaterlingPeterson = self.check_easterlingPeterson_discontinuity(df)
        return linReg, mannKendall, eaterlingPeterson


class Mast:
    def __init__(self, mast_id: str, lat: int, lon: int, elevation: float):
        self.id: str = mast_id
        self.lat: int = lat
        self.lon: int = lon
        self.elevation: float = elevation
        easting, northing, zone, letter = from_latlon(lat, lon)
        self.easting: float = easting
        self.northing: float = northing
        self.utm_zone: int = zone
        self.magnetic_declination = geomag.declination(lat, lon)
        self.MastData: MastData = None

    def has_data(self) -> bool:
        return self.MastData is not None

    def add_data(self, filepath: str, data_key: DataKey = None, source: str = "windog", time_shift: int = 0):
        self.MastData = MastData(filepath, data_key, source, time_shift)
        print("Data added to mast: " + self.id + "!")


def check_combinedData(masts: list, are_masts: bool = True) -> list:
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
    utmZones = []
    for mast in masts:
        ids.append(mast.id)
        direction[mast.id] = mast.MastData.df[mast.MastData.primary_windDirection_column] * 1
        direction[mast.id][(direction[mast.id] < 60) | (direction[mast.id] > 360)] = np.nan
        utmZones.append(mast.utm_zone)
    if len(set(utmZones)) > 1:
        utmZone = min(utmZones)
        for mast in masts:
            if mast.utm_zone > utmZone:
                mast.utm_zone = utmZone
                easting, northing, zone, letter = from_latlon(mast.lat, mast.lon, force_zone_number=utmZone)
                mast.easting = easting
                mast.northing = northing
    direction.dropna(inplace=True)
    dir_corr = direction.corr()
    slopes = pd.DataFrame(index=ids, columns=ids)
    offsets = pd.DataFrame(index=ids, columns=ids)
    for mast1 in my_masts:
        for mast2 in my_masts:
            if mast1.id != mast2.id:
                slope, offset = correlate_LLS(direction, mast1.id, mast2.id)
                slopes[mast1.id][mast2.id] = slope
                offsets[mast1.id][mast2.id] = offset
    if are_masts:
        if dir_corr.min().min() < 0.9:
            warnings.append(
                "Correlation coefficient between the primary direction sensor on each mast is < 0.9. This may indicate a time shift in the input data. Please check."
            )
        if offsets.abs().max().max() > 20:
            warnings.append(
                "The offset from the correlation between the primary direction sensor on each mast is > 20. This may indicate a miscalibration in one or more direction sector. Please check."
            )
    else:
        if dir_corr.min().min() < 0.9:
            warnings.append(
                "Correlation coefficient between the primary direction sensor on each long-term source is < 0.9. This may indicate a time shift in the input data. Please check."
            )
    return warnings


def apply_global_PC(time_series: pd.Series, p_curve: pd.DataFrame, prod_col: str):
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


def longTerm_selectionCriteria(mast: Mast, lt_sources: list) -> pd.DataFrame:
    """
    Condense list of long-term data sources to a common period and compare sources to
    a reference mast

    Parameters
    ----------
    mast : Mast
        reference mast used for comparison to potential long-term data sources.
    lt_sources : list
        list of Mast objects representing long term data sources.

    Returns
    -------
    selection : pd.DataFrame
        Summary statistics describing the relationship between the reference mast and long-term source.

    """
    # select common period
    end = min([i.MastData.data_end for i in lt_sources])
    end = pd.Timestamp(end.year, end.month, 1)  # make it start on the 1st of the month
    possible_start = max([i.MastData.data_start for i in lt_sources])
    if possible_start.year >= 2000:
        if pd.Timestamp(possible_start.year, end.month, 1) > possible_start:
            start = pd.Timestamp(possible_start.year, end.month, 1)
        else:
            start = pd.Timestamp(possible_start.year + 1, end.month, 1)
    else:
        start = pd.Timestamp(2000, end.month, 1)
    selection = pd.DataFrame(
        index=[
            "R2",
            "Weibull A Mast",
            "Weibull A Long Term",
            "Weibull A bias",
            "Weibull k Mast",
            "Weibul k Long Term",
            "Weibull k bias",
            "Wind rose",
            "Trend",
            "Discontinuity",
        ]
    )
    # update data and check for trends
    for source in lt_sources:
        # trim data
        source.MastData.df = source.MastData.df[(source.MastData.df.index >= start) & (source.MastData.df.index < end)]
        # update weibulls & directional frequency
        source.MastData._get_weibullFit()
        source.MastData._get_directionalData()
        # check for trends and discontinuity
        linReg, mannKendall, easterlingPeterson = source.MastData.verify_longTerm_viability(
            source.MastData.df[source.MastData.primary_windSpeed_column]
        )
        trend = False
        if (linReg) and (mannKendall):
            trend = True
        selection[source.id] = [
            # R2
            mast.MastData.longTerm_corr[source.id]["R2"],
            # Weibull A comparison
            mast.MastData.A[mast.MastData.primary_windSpeed_column],
            source.MastData.A[source.MastData.primary_windSpeed_column],
            source.MastData.A[source.MastData.primary_windSpeed_column]
            / mast.MastData.A[mast.MastData.primary_windSpeed_column]
            - 1,
            # weibul k comparison
            mast.MastData.k[mast.MastData.primary_windSpeed_column],
            source.MastData.k[source.MastData.primary_windSpeed_column],
            source.MastData.k[source.MastData.primary_windSpeed_column]
            / mast.MastData.k[mast.MastData.primary_windSpeed_column]
            - 1,
            # wind rose comparison
            abs(
                mast.MastData.directional_WindSpeed["frequency"] - source.MastData.directional_WindSpeed["frequency"]
            ).sum()
            / 2,
            # trends & discontinuity
            trend,
            easterlingPeterson,
        ]
    return selection


def longTerm_correlations(masts: list, lt_sources: list, corr_type: Callable, lt_dur: str = "W", mast_dur: str = "D"):
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


def create_heatMap(df: pd.DataFrame):
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
    figPath : TYPE
        DESCRIPTION.

    """
    x_mean = (df["easting"].max() - df["easting"].min()) / 2 + df["easting"].min()
    y_mean = (df["northing"].max() - df["northing"].min()) / 2 + df["northing"].min()
    buffer = (
        max([(df["easting"].max() - df["easting"].min()), (df["northing"].max() - df["northing"].min())]) / 2 * 1.03
    )
    plt.plot(figsize=(6.4, 6.4))
    plt.xlim([x_mean - buffer, x_mean + buffer])
    plt.ylim([y_mean - buffer, y_mean + buffer])
    nx, ny = np.mgrid[x_mean - buffer : x_mean + buffer : 1000, y_mean - buffer : y_mean + buffer : 1000]
    h = [(i["easting"], i["northing"]) for ct, i in df.iterrows()]
    nz = griddata(h, df["wind speed"].values, (nx, ny), method="linear")
    plt.contour(nx, ny, nz, linewidths=0.5)
    plt.contourf(nx, ny, nz)
    plt.scatter(df["easting"], df["northing"], s=7, c="k")
    for ind, label in df.iterrows():
        plt.annotate(ind, (label["easting"] + buffer * 0.03, label["northing"] + buffer * 0.03))
    figPath = os.path.join(os.path.abspath(os.path.dirname(__file__)), "figs", "WSHeatMap.png")
    plt.savefig(figPath, bbox_inches="tight")
    plt.close()
    return figPath


def run_concurrentPeriod(
    masts: list, powerCurve: pd.DataFrame, prod_col: str = "prod", referenceMast: Optional[str] = None
) -> (pd.DataFrame, int):
    """
    Return summary of mast concurrent period

    Parameters
    ----------
    masts : list
        list of Mast objects.
    powerCurve : pd.DataFrame
        data frame representing turbine power curve
    prod_col : str
        name of column in powerCurve data frame representing power. The default is 'prod'
    referenceMast : str
        mast id of the selected reference mast. The default is None

    Returns
    -------
    df_concurrentPeriod : pd.DataFrame
        dataframe containing summary of concurrent period analysis.
    max_length : int
        maximum length of a gap filled data set without data extension (for use in mcp process)
    fig : list
        list of paths to figures generated in the concurrent process

    """
    ratedPower = powerCurve[prod_col].max()
    # set up concurrent period results table
    df_concurrentPeriod = pd.DataFrame(index=[i.id for i in masts])
    df_concurrentPeriod["Measurement Height (m)"] = [i.MastData.primary_windSpeed_column_ht for i in masts]
    # compile concurrent wind speed data
    df_heatMap = pd.DataFrame(
        {
            "northing": [i.northing for i in masts],
            "easting": [i.easting for i in masts],
            "wind speed": [np.nan for i in masts],
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
    df_heatMap["wind speed"] = cp.mean()
    fig = create_heatMap(df_heatMap)
    # save concurrent period wind speeds
    length_concurrentPeriod = len(cp)
    df_concurrentPeriod["valid time steps"] = length_concurrentPeriod
    df_concurrentPeriod["mean wind speed"] = cp.mean().T
    if referenceMast is not None:
        fig = [fig] + [os.path.join(os.path.abspath(os.path.dirname(__file__)), "figs", "WSRatio.png")]
        # save values for energy content comparison
        scale = df_concurrentPeriod["mean wind speed"][referenceMast] / df_concurrentPeriod["mean wind speed"]
        plt.plot(range(len(scale)), 1 / scale, color=my_colors[0])
        plt.axhline(y=1.0, linestyle=":", color="k")
        plt.xticks(range(len(scale)), cp.columns)
        plt.ylabel("Wind speed ratio relative to " + referenceMast)
        plt.savefig(fig[-1])
        plt.close()
    # initiate next columns
    df_concurrentPeriod["zero power"] = np.nan
    df_concurrentPeriod["rated power"] = np.nan
    df_concurrentPeriod["AEP"] = np.nan
    df_concurrentPeriod["NCF"] = np.nan
    df_concurrentPeriod["scaled wind speed"] = df_concurrentPeriod["mean wind speed"] * scale
    df_concurrentPeriod["scaled AEP"] = np.nan
    df_concurrentPeriod["scaled NCF"] = np.nan
    for mast in cp:
        # use power curve to convert wind speeds to production
        production = apply_global_PC(cp[mast], powerCurve, prod_col)
        # count non-production
        df_concurrentPeriod["zero power"][mast] = production[production == 0].size / length_concurrentPeriod
        # count rated production
        df_concurrentPeriod["rated power"][mast] = production[production == ratedPower].size / length_concurrentPeriod
        # save production statistics
        df_concurrentPeriod["AEP"][mast] = production.sum()
        df_concurrentPeriod["NCF"][mast] = df_concurrentPeriod["AEP"][mast] / (ratedPower * length_concurrentPeriod)
        # scale wind speeds and reapply power curve for energy content comparison
        production = apply_global_PC(cp[mast] * scale[mast], powerCurve, prod_col)
        df_concurrentPeriod["scaled AEP"][mast] = production.sum()
        df_concurrentPeriod["scaled NCF"][mast] = df_concurrentPeriod["scaled AEP"][mast] / (
            ratedPower * length_concurrentPeriod
        )
    return df_concurrentPeriod, max_length_df, fig


def correlate_windSpeed_sensors_10m(masts: list) -> pd.DataFrame:
    """
    Return mast to mast coefficients of determination without reaveraging data

    Parameters
    ----------
    masts : list
        list of Mast objects.

    Returns
    -------
    pd.DataFrame
        dataframe containing coefficients of determination between each mast and
        wind speed sensor.

    """
    df = pd.DataFrame()
    for mast in masts:
        for col in mast.MastData.data_key.windSpeed_columns.values():
            df[mast.id + "_" + col] = mast.MastData.df[col]
    return pow(df.corr(), 2)


def correlate_TLS(df: pd.DataFrame, col_x: str, col_y: str) -> (float, float):
    """
    run total least square correlation between two columns of a dataframe

    Parameters
    ----------
    df : pd.DataFrame
        data frame containing two columns you wish to correlate.
    col_x : str
        column of data frame representing the independent variable.
    col_y : str
        column of data frame representing the dependent variable.

    Returns
    -------
    (float, float)
        slope and intercept of total least squares correlation.

    """
    s_xx = pow(df[col_x] - df[col_x].mean(), 2).sum()
    s_yy = pow(df[col_y] - df[col_y].mean(), 2).sum()
    s_xy = ((df[col_x] - df[col_x].mean()) * (df[col_y] - df[col_y].mean())).sum()
    slope = (s_yy - s_xx + pow(pow(s_xx - s_yy, 2) + 4 * pow(s_xy, 2), 0.5)) / (2 * s_xy)
    intercept = df[col_y].mean() - slope * df[col_x].mean()
    return slope, intercept


def correlate_LLS(df: pd.DataFrame, col_x: str, col_y: str) -> (float, float):
    """
    run linear least square correlation between two columns of a dataframe

    Parameters
    ----------
    df : pd.DataFrame
        data frame containing two columns you wish to correlate.
    col_x : str
        column of data frame representing the independent variable.
    col_y : str
        column of data frame representing the dependent variable.

    Returns
    -------
    (float, float)
        slope and intercept of linear least squares correlation.

    """
    linear_regressor = LinearRegression()
    reg = linear_regressor.fit(df[col_x].values.reshape(-1, 1), df[col_y].values.reshape(-1, 1))
    return reg.coef_[0][0], reg.intercept_[0]


def MCP_gapFill(
    masts: list,
    df_correlation: pd.DataFrame,
    mcp_method: Callable,
    stop_length: int,
    min_r2: float = 0.5,
    cutoff_WS: float = 3,
    extend_data: bool = False,
):
    """
    gap fill wind speed columns in Mast.MastData object, using the best available
    correlation

    Parameters
    ----------
    masts : list
        list of Masts containing MastData.
    df_correlation : pd.DataFrame
        data frame containing coefficients of determination between each mast and sensor.
    mcp_method : Callable
        function for correlation method (eg. correlate_TLS).
    stop_length : int
        the maximum length of the final dataset. Used to stop gap filling of a
        mast once maximum data recovery has been reached.
    min_r2 : float, optional
        minimum coefficient of determination for gap filling to occur. The default is 0.5.
    cutoff_WS : float, optional
        minimum wind speed used in sensor to sensor correlations. The default is 3.
    extend_data : bool, optional
        if True, extend each data set to the same starting and ending points.
        If False, gap fill data but do not extend past original time frame.
        The default is False.

    Returns
    -------
    df_gapFill : pd.DataFrame
        data frame containing a summary of changes to each sensor as a result
        of the gap filling process.

    """
    # convert coefficients of determination to rank values
    df_correlation[df_correlation < min_r2] = np.nan
    df_correlation = df_correlation.rank(ascending=False)
    correlation_options = len(df_correlation)
    mast_map = {mast.id: index for index, mast in enumerate(masts)}
    df_gapFill = pd.DataFrame(
        index=[
            ("Valid Data Points", "Before"),
            ("Valid Data Points", "After"),
            ("Valid Data Points", "Change"),
            ("DRR", "Before"),
            ("DRR", "After"),
            ("DRR", "Change"),
            ("Mean", "Before"),
            ("Mean", "After"),
            ("Mean", "Change"),
            ("Min", "Before"),
            ("Min", "After"),
            ("Min", "Change"),
            ("Max", "Before"),
            ("Max", "After"),
            ("Max", "Change"),
            ("Weibull k", "Before"),
            ("Weibull k", "After"),
            ("Weibull k", "Change"),
        ]
    )
    for mast in masts:
        values = [np.nan for i in range(18)]
        for col in mast.MastData.data_key.windSpeed_columns.values():
            mast.MastData.df[col + "_filled"] = mast.MastData.df[col] * 1
            values[0] = len(mast.MastData.df[col][~mast.MastData.df[col].isna()])
            values[3] = values[0] / len(mast.MastData.df)
            values[6] = mast.MastData.df[col].mean()
            values[9] = mast.MastData.df[col].min()
            values[12] = mast.MastData.df[col].max()
            values[15] = mast.MastData.k[col]
            # ranks other sensors in order of best match
            corr_rank = {value - 1: column for column, value in df_correlation[mast.id + "_" + col].iteritems()}
            for rank in range(1, correlation_options):
                if corr_rank.get(rank):
                    # if rank exists (ie. has r2 >= min_r2), run correlation and add data
                    corr_mast = masts[mast_map[corr_rank[rank][: corr_rank[rank].index("_")]]]
                    corr_col = corr_rank[rank][corr_rank[rank].index("_") + 1 :]
                    mast.MastData.df["corr_x"] = corr_mast.MastData.df[corr_col] * 1
                    mast.MastData.df["dir_x"] = corr_mast.MastData.df["directionSector"] * 1
                    # save space for synthesized data
                    mast.MastData.df["synthesized"] = np.nan
                    # determine directional correlations
                    for direction in range(mast.MastData.directionSectors):
                        df_sub = mast.MastData.df[[col, "corr_x"]][mast.MastData.df["dir_x"] == direction]
                        # remove low wind speed values
                        df_sub[df_sub < cutoff_WS] = np.nan
                        # remove nan values prior to correlation
                        df_sub.dropna(inplace=True)
                        if len(df_sub) > 0:
                            # don't bother running the correlation if there are no data points
                            slope, offset = mcp_method(df_sub, "corr_x", col)
                            # apply directional correlations
                            mast.MastData.df["synthesized"][mast.MastData.df["dir_x"] == direction] = (
                                mast.MastData.df["corr_x"][mast.MastData.df["dir_x"] == direction] * slope + offset
                            )
                    # fill empty time stamps
                    mast.MastData.df[col + "_filled"].fillna(mast.MastData.df["synthesized"], inplace=True)
                    if len(mast.MastData.df[~mast.MastData.df[col + "_filled"].isna()]) == stop_length:
                        break
                else:
                    # stop if you're out of columns with sufficiently high r2 value
                    break
            values[1] = len(mast.MastData.df[col + "_filled"][~mast.MastData.df[col + "_filled"].isna()])
            values[4] = values[1] / len(mast.MastData.df)
            values[7] = mast.MastData.df[col + "_filled"].mean()
            values[10] = mast.MastData.df[col + "_filled"].min()
            values[13] = mast.MastData.df[col + "_filled"].max()
            df_gapFill[(mast.id, col)] = values
        # remove placeholder column
        mast.MastData.df.drop(columns=["synthesized", "corr_x", "dir_x"], inplace=True)
        if not extend_data:
            # remove extended data if unwanted
            mast.MastData.df = mast.MastData.df[mast.MastData.df.index >= mast.MastData.data_start]
            mast.MastData.df = mast.MastData.df[mast.MastData.df.index <= mast.MastData.data_end]
    # run cleanup to avoid bogging down code with unnecessary info
    post_k = []
    for mast in masts:
        for col in mast.MastData.data_key.windSpeed_columns.values():
            # drop unfilled data
            mast.MastData.df.drop(columns=[col], inplace=True)
            # rename _filled columns
            mast.MastData.df.rename(columns={col + "_filled": col}, inplace=True)
        mast.MastData._get_weibullFit()
        for col in mast.MastData.data_key.windSpeed_columns.values():
            post_k.append(mast.MastData.k[col])
    df_gapFill = df_gapFill.T
    df_gapFill[("Weibull k", "After")] = post_k
    for i in range(6):
        df_gapFill[df_gapFill.columns[3 * i + 2]] = (
            df_gapFill[df_gapFill.columns[3 * i + 1]] - df_gapFill[df_gapFill.columns[3 * i]]
        )
    return df_gapFill


def get_referencePeriods(masts: list):
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
    refPeriod_options = pd.DataFrame(index=["start", "end", "possible data points", "length", "recovery rate", "MAE"])
    ct = 0
    for years in range(max_years, 0, -1):
        for start_year in range(max_years - years + 1):
            for month in range(12):
                start_month = ref_month + month
                if start_month > 12:
                    start_month -= 12
                start = pd.Timestamp(ref_year + start_year, start_month, 1)
                end = pd.Timestamp(ref_year + start_year + years, start_month, 1)
                possible = int((end - start).total_seconds() / 600)
                refPeriod_options[ct] = [start, end, possible, years, np.nan, 0]
                ct += 1

    # pull primary met mast data
    refPeriod_options = refPeriod_options.T
    masts = pd.DataFrame(
        [i.MastData.df[i.MastData.primary_windSpeed_column] for i in masts], index=[i.id for i in masts]
    ).T

    # asses the reference period options
    sumData_cols = []
    mae_cols = []
    for mast in masts:
        valid_points = []
        mean_ws = []
        for index, refPeriod in refPeriod_options.iterrows():
            subset = masts[mast][(masts.index >= refPeriod["start"]) & (masts.index < refPeriod["end"])].dropna()
            valid_points.append(subset.size)
            mean_ws.append(subset.mean())
        refPeriod_options[mast + " valid data"] = valid_points
        refPeriod_options[mast + " recovery rate"] = (
            refPeriod_options[mast + " valid data"] / refPeriod_options["possible data points"]
        )
        refPeriod_options[mast + " mean wind speed"] = mean_ws
        refPeriod_options[mast + " MAE"] = refPeriod_options[mast + " mean wind speed"] - masts[mast].mean()
        # save data point columns
        sumData_cols.append(mast + " valid data")
        mae_cols.append(mast + " MAE")

    # complete nan columns
    refPeriod_options["recovery rate"] = refPeriod_options[sumData_cols].sum(axis=1) / (
        refPeriod_options["possible data points"] * len(masts.columns)
    )
    for data, mae in zip(sumData_cols, mae_cols):
        refPeriod_options["MAE"] += (
            refPeriod_options[data] * refPeriod_options[mae].replace(np.nan, 0)
        ) / refPeriod_options["possible data points"]

    # select pareto points for recovery rate & mae fro each number of years
    refPeriod_options["pareto"] = 0
    max_recovery = {i: [0, 0] for i in range(max_years, 0, -1)}
    for years in range(max_years, 0, -1):
        # for each number of years
        df_sub = refPeriod_options[refPeriod_options["length"] == years]
        for index, row in df_sub.iterrows():
            # find points on the pareto frontier
            if len(df_sub[(df_sub["MAE"] < row["MAE"]) & (df_sub["recovery rate"] > row["recovery rate"])]) == 0:
                refPeriod_options["pareto"][index] = 1
                if row["recovery rate"] > max_recovery[years][1]:
                    max_recovery[years] = [index, row["recovery rate"]]
    recovery_min = max([value[1] for key, value in max_recovery.items()]) - 0.03
    max_recovery = {key: value for key, value in max_recovery.items() if value[1] >= recovery_min}
    default_period = max_recovery[max(max_recovery.keys())][0]
    return refPeriod_options[refPeriod_options["pareto"] == 1].drop(columns=["pareto"]), default_period


def cut_reference_period(masts: list, start: pd.Timestamp, end: pd.Timestamp):
    """
    cut Mast.MastData.df object to the referenece period

    Parameters
    ----------
    masts : list
        list of Mast objects.
    start : pd.Timestamp
        start of data period.
    end : pd.Timestamp
        end of data period.

    Returns
    -------
    None.

    """
    for mast in masts:
        mast.MastData.df = mast.MastData.df[(mast.MastData.df.index >= start) & (mast.MastData.df.index < end)]
        mast.MastData._get_weibullFit()
        mast.MastData._get_directionalData()


def calculate_shear(masts: list, max_veer: float = 100, min_WS: float = 3):
    """
    calculate directional and weighted directional (global) shear values for each
    Mast.MastData object

    Parameters
    ----------
    masts : list
        list of Mast objects.
    max_veer : float, optional
        maximum wind direction veer present in a time stamp used in determining
        the shear parameter. The default is 100.
    min_WS : float, optional
        minimum wind speed present in a time stamp used in determining
        the shear parameter. The default is 3.

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
            sub_mast_data[col][sub_mast_data[col] < min_WS] = np.nan
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


def plot_monthlyWS(masts: list, concurrent: bool = False):
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
    figPath = os.path.join(os.path.abspath(os.path.dirname(__file__)), "figs", "MonthlyWS" + str(ct) + ".png")
    plt.savefig(figPath, bbox_inches="tight")
    plt.close()
    return figPath


def plot_diurnalWS(masts: list, concurrent: bool = False):
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
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "figs", "DirunalWS" + str(ct) + ".png")
    ):
        ct += 1
    figPath = os.path.join(os.path.abspath(os.path.dirname(__file__)), "figs", "DirunalWS" + str(ct) + ".png")
    plt.savefig(figPath, bbox_inches="tight")
    plt.close()
    return figPath


def plot_WSDist(masts: list, concurrent: bool = False):
    """
    plot wind speed distribution of primary wind speed sensor

    Parameters
    ----------
    masts : list
        list of Mast objects.
    concurrent : bool, optional
        if True, calculate wind speed distributions using only concurrent period data.
        The default is False.

    Returns
    -------
    figPath : str
        path to the resulting wind speed distribution plot.

    """
    x = [i + 0.5 for i in range(25)]
    plt.figure(figsize=(10, 6.4))
    for ct, mast in enumerate(masts):
        if concurrent:
            y = [
                len(
                    mast.MastData.df[
                        (mast.MastData.df[mast.MastData.primary_windSpeed_column] < i + 1)
                        & (mast.MastData.df[mast.MastData.primary_windSpeed_column] >= i)
                        & (mast.MastData.df["CP"] == 1)
                    ]
                )
                for i in range(25)
            ]
        else:
            y = [
                len(
                    mast.MastData.df[
                        (mast.MastData.df[mast.MastData.primary_windSpeed_column] < i + 1)
                        & (mast.MastData.df[mast.MastData.primary_windSpeed_column] >= i)
                    ]
                )
                for i in range(25)
            ]
        y = [i / sum(y) for i in y]
        plt.plot(x, y, color=my_colors[ct], label=mast.id + ": " + str(mast.MastData.primary_windSpeed_column_ht) + "m")
    plt.legend(loc="upper center", bbox_to_anchor=(0, -0.1, 1, 0), ncol=4)
    plt.title("Wind speed distribution")
    plt.xlabel("Wind Speed")
    plt.ylabel("Frequency")
    plt.xlim([0, 25])
    ct = 0
    while os.path.exists(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "figs", "WSDistribution" + str(ct) + ".png")
    ):
        ct += 1
    figPath = os.path.join(os.path.abspath(os.path.dirname(__file__)), "figs", "WSDistribution" + str(ct) + ".png")
    plt.savefig(figPath, bbox_inches="tight")
    plt.close()
    return figPath


def plot_windRose(masts: list, concurrent: bool = False):
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
    figPath = os.path.join(os.path.abspath(os.path.dirname(__file__)), "figs", "windRose" + str(ct) + ".png")
    plt.savefig(figPath, bbox_inches="tight")
    plt.close()
    return figPath


def basic_quadPlots(masts: list, concurrent=False):
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
    figs = []
    figs.append(plot_monthlyWS(masts, concurrent))
    figs.append(plot_diurnalWS(masts, concurrent))
    figs.append(plot_WSDist(masts, concurrent))
    figs.append(plot_windRose(masts, concurrent))
    return figs


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
    output_filepath = r"test\testWRA.xlsx"
    """ Initiate MET001 """
    mast_id = "MET001"
    latitude = 25.0
    longitude = -120.0
    elevation = 1421
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
    data_key = DataKey(wind_speed, wind_direction, temperature, relative_humidity, pressure)
    MET001 = Mast(mast_id, latitude, longitude, elevation)
    MET001.add_data(input_file, data_key)

    """ Initiate MET002 """
    mast_id = "MET002"
    latitude = 25.1
    longitude = -120.1
    elevation = 1421
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
    data_key = DataKey(wind_speed, wind_direction, temperature, relative_humidity, pressure)
    MET002 = Mast(mast_id, latitude, longitude, elevation)
    MET002.add_data(input_file, data_key)

    """ set masts and reference mast """
    my_masts = [MET001, MET002]
    reference_mast = MET001

    """ select whether gaps should be filled (True) or left (false) """
    gap_filling = False

    """ Set Power Curve for energy content comparisons """
    powerCurve = pd.read_csv(r"test\turbinePC.csv", index_col=0)

    """ Input Long-term data """
    source_id = "vortex"
    latitude = 45.0
    longitude = -120.0
    input_file = r"test\test_vortexReader.txt"
    vortex = Mast(source_id, latitude, longitude, None)
    vortex.add_data(input_file, source="vortex")

    source_id = "era5"
    latitude = 45.0
    longitude = -120.0
    input_file = r"test\test_windographerReader_ERA5.txt"
    era5 = Mast(source_id, latitude, longitude, None)
    era5.add_data(input_file, source="windog_download", time_shift=-6)

    source_id = "merra2"
    latitude = 45.0
    longitude = -120.0
    input_file = r"test\test_windographerReader_MERRA2.txt"
    merra2 = Mast(source_id, latitude, longitude, None)
    merra2.add_data(input_file, source="windog_download", time_shift=-6)
    longTerm_sources = [vortex, era5, merra2]

    """ run WRA! """
    start = datetime.now()
    warnings = check_combinedData(my_masts, are_masts=True)
    warnings.append(check_combinedData(my_masts + longTerm_sources, are_masts=False))
    try:
        os.mkdir(os.path.join(os.path.abspath(os.path.dirname(__file__)), "figs"))
    except:
        pass
    # TODO: check long-term data and delete what is only getting in the way
    longTerm_correlations(my_masts, longTerm_sources, correlate_TLS)
    # run concurrent period calc after all masts created
    df_concurrent, mcp_length, fig = run_concurrentPeriod(my_masts, powerCurve, referenceMast=reference_mast.id)
    # run MCP after all masts created
    df_correlation = correlate_windSpeed_sensors_10m(my_masts)
    if gap_filling:
        df_gapFill = MCP_gapFill(my_masts, df_correlation, mcp_method=correlate_TLS, stop_length=mcp_length)
    # put in ref period selector
    refPeriod_options, default_selection = get_referencePeriods(my_masts)
    for mast in my_masts:
        mast.MastData.refPeriod_start = refPeriod_options["start"][default_selection]
        mast.MastData.refPeriod_end = refPeriod_options["end"][default_selection]
    cut_reference_period(
        my_masts, refPeriod_options["start"][default_selection], refPeriod_options["end"][default_selection]
    )
    # TODO: user input here to select reference period or accept default (PLACE INTO 5a tab)
    # TODO: add in reference mast selection
    lt_ws = [i.MastData.df[i.MastData.primary_windSpeed_column].mean() for i in longTerm_sources]
    df_longTerm_selection = longTerm_selectionCriteria(reference_mast, longTerm_sources)
    calculate_shear(my_masts)
    """ print warnings from code """
    for warning in warnings:
        print(warning)
    try:
        os.remove(os.path.join(os.path.abspath(os.path.dirname(__file__)), "figs"))
    except:
        print("Could not remove figure folder")
    print("------------------")
    print()
    print("Code completed in: " + str((datetime.now() - start).total_seconds() / 60) + " minutes!")
    print()
    print("------------------")
