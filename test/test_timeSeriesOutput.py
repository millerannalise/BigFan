# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27, 2022

@author: Annalise Miller
"""

import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import timeSeriesOutput as tso


def test_read_windog_custom_file():
    # test custom windographer input (measured data)
    df_test = pd.DataFrame(
        [
            [0, 25, 1013.25, 7, 10, 6],
            [0, 25, 1013.25, 7, 10, 6],
            [0, 25, 1013.25, 7, 10, 6]
        ],
        index=[pd.Timestamp(2020, 1, 1, 0, 0), pd.Timestamp(2020, 1, 1, 0, 10), pd.Timestamp(2020, 1, 1, 0, 20)],
        columns=['Direction', 'Temperature', 'Pressure', 'Speed1', 'Speed1SD', 'Speed2']
    )
    assert all(tso.read_windog_file('test_windographerReader_custom.txt') == df_test)


def test_read_windog_era5_file():
    # test standardized format from windographer data downloader using ERA5
    df_test = pd.DataFrame(
        [
            [7., 0., 25., 1013.25],
            [7., 0., 25., 1013.25],
            [7., 0., 25., 1013.25]
        ],
        index=[pd.Timestamp(1979, 1, 1, 0, 0), pd.Timestamp(1979, 1, 1, 1, 0), pd.Timestamp(1979, 1, 1, 2, 0)],
        columns=['WS', 'WD', 'Temp', 'Pressure']
    )
    assert all(tso.read_windog_file('test_windographerReader_ERA5.txt', separator='\t') == df_test)


def test_read_windog_merra2_file():
    # test standardized format from windographer data downloader using MERRA2
    df_test = pd.DataFrame(
        [
            [7., 0., 25., 1013.25],
            [7., 0., 25., 1013.25],
            [7., 0., 25., 1013.25]
        ],
        index=[pd.Timestamp(1979, 1, 1, 0, 0), pd.Timestamp(1979, 1, 1, 1, 0), pd.Timestamp(1979, 1, 1, 2, 0)],
        columns=['WS', 'WD', 'Temp', 'Pressure']
    )
    assert all(tso.read_windog_file('test_windographerReader_MERRA2.txt', separator='\t') == df_test)


def test_read_vortex_file():
    # test standardized format from vortex output
    df_test = pd.DataFrame(
        [
            [7., 0., 25., 1.225, 1013.25, 3.1, 50.],
            [7., 0., 25., 1.225, 1013.25, 3.1, 50.],
            [7., 0., 25., 1.225, 1013.25, 3.1, 50.]
        ],
        index=[pd.Timestamp(2020, 1, 1, 0, 0), pd.Timestamp(2020, 1, 1, 1, 0), pd.Timestamp(2020, 1, 1, 2, 0)],
        columns=['WS', 'WD', 'Temp', 'Rho', 'Pressure', 'RiNumber', 'RH(%)']
    )
    assert all(tso.read_vortex_file('test_vortexReader.txt', separator='\t') == df_test)


df_testCSV = pd.read_csv(
    'df_testCSV.csv',
    index_col=0,
    parse_dates=True,
    infer_datetime_format=True)

test_parameters = {
    'InputFiles': {
        # label for source: (path to source, source format)
        0: (r'C:\Users\Annalise\Downloads\test_windographerReader_custom.txt', 'windographer'),
        1: (r'C:\Users\Annalise\Downloads\test_vortexReader.txt', 'vortex'),
    },
    'MastPrimarySource': 0,
    'MastName': 'MET001',  # str
    'Latitude': 45.,  # float
    'Longitude': -120.,  # float
    'TimeZone': -8,  # int
    'HighSensorName': 'Speed1',  # str
    'HighSensorHeight': 60,  # float
    'LongTermWindSpeed': 9.5,  # float
    'ShearValue': 0.2,  # float
    'HighSensorStd': 'Speed1SD',  # str
    'LowSensorName': 'Speed2',  # str
    'LowSensorHeight': 30,  # float
    'DesiredHeights': [100],  # list of floats
    'WindDirectionName': 'Direction',
    'TemperatureName': (1, '12x24'),
    'PressureName': (1, 'TS'),
    'TargetAirDensity': 1.225,  # float
    'HubHeightTargetAirDensity': 100,  # float like
    'gap_fill_WS_WD': True,  # bool
    'OutputLocation': r'C:\Users\Annalise\Downloads'  # str
}

df_random = pd.DataFrame(
    np.random.rand(len(df_testCSV), 1),
    columns=['random test'],
    index=df_testCSV.index)


def test_create_12x24_12x24Output():
    # test create 12x24 with 12x24 output
    df_test = pd.DataFrame(
        index=[i for i in range(24)],
        columns=[i + 1 for i in range(12)]
    )
    for i in range(12):
        df_test[i + 1] = i + 1
    assert all(tso.create_12x24(df_testCSV, 'WS', output='12x24') == df_test)


def test_create_12x24_columnOutput():
    # test create 12x24 with column output
    df_test = df_testCSV * 1.
    df_test['12x24'] = df_test.index.month
    assert all(tso.create_12x24(df_testCSV, 'WS', output='column') == df_test)


def test_fill_with_12x24():
    # test function for gap filling with data from a 12x24
    assert all(tso.fill_with_12x24(df_testCSV, 'WS GAP')['WS GAP'] == df_testCSV['WS'])


def test_create_with_12x24():
    df_test = df_testCSV * 1.
    df_test['12x24'] = df_test.index.month
    assert all(tso.create_with_12x24(df_testCSV, '12x24', df_testCSV, 'WS') == df_testCSV['WS'])


def test_append_values():
    # test appending values from a different data frame
    assert all(tso.append_values(df_testCSV, 'test', df_random, 'random test')['test'] == df_random['random test'])


def test_calc_AirDensity():
    # test air density calculation
    # inputs df, parameters