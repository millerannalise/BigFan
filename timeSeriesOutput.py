# -*- coding: utf-8 -*-
"""
Created on Fri May 20, 2022

@author: Annalise Miller
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def read_windog_file(file_path, separator=',', windog_tz=None):
    '''
    read in a windographer file. If the file is in the ERA5 or MERRA2 format
    from the windographer data downloader, reformat columns as required

    Parameters
    ----------
    file_path : str
        path to file with windographer output format
    windog_tz : Int, optional
        time adjustment for incorrect windographer timestamps. The default is None.

    Returns
    -------
    df : pd.DataFrame
        data frame of windographer time-series

    '''
    with open(file_path, newline='') as my_file:
        all_lines = my_file.readlines()
    for ct, line in enumerate(all_lines):
        if all_lines[ct][:9].lower() in ['date/time']:
            save_start = ct + 1
            cols = all_lines[ct].encode('utf-8', errors='ignore').decode()
            cols = cols.strip('\r\n').split(separator)
            break
    try:
        df = pd.read_csv(file_path,
                         skiprows=save_start,
                         names=cols,
                         index_col=0,
                         sep=separator,
                         parse_dates=True,
                         infer_datetime_format=True,
                         encoding_errors='ignore')
    except TypeError:
            df = pd.read_csv(file_path,
                     skiprows=save_start,
                     names=cols,
                     index_col=0,
                     sep=separator,
                     parse_dates=True,
                     infer_datetime_format=True)
    if windog_tz is not None:
        df.index = [i + pd.Timedelta(hours=windog_tz) for i in df.index]
    df = df[[k for k in df.keys() if not all(pd.isna(df[k]))]]
    if 'ERA5' in file_path:
        try:
            df.rename(
                columns={
                    'Speed_100m [m/s]': 'WS',
                    'Direction_100m [degrees]': 'WD',
                    'Temperature_2m [degrees C]': 'Temp',
                    'Pressure_0m [kPa]': 'Pressure',
                    }, inplace=True
                )
        except:
            pass
    if 'MERRA2' in file_path:
        try:
            df.rename(
                columns={
                    'Speed_50m [m/s]': 'WS',
                    'Direction_50m [degrees]': 'WD',
                    'Temperature_10m [degrees C]': 'Temp',
                    'Pressure_0m [kPa]': 'Pressure',
                    # 'De(k/m3)': 'Rho',
                    }, inplace=True
                )
        except:
            pass
    return df.astype(float)


def read_vortex_file(file_path, separator=None):
    """
    read in data in vortex output format.

    Parameters
    ----------
    file_path : TYPE
        path to file with vortex output format.
    separator : TYPE, optional
        column separator in vortex file format. The default is None.

    Returns
    -------
    pd.DataFrame
        data frame of time series input data.

    """
    with open(file_path, newline='') as my_file:
        all_lines = my_file.readlines()
    for ct, line in enumerate(all_lines):
        if all_lines[ct][:8] in ['YYYYMMDD']:
            save_start = ct + 1
            cols = all_lines[ct].encode('utf-8', errors='ignore').decode()
            cols = cols.strip('\r\n').split(separator)
            break
    all_lines = [i.strip('\r\n').split(separator) for i in all_lines[save_start:]]
    df = pd.DataFrame(all_lines, columns=cols)
    df.index = pd.to_datetime(df['YYYYMMDD'] + df['HHMM'], format='%Y%m%d%H%M')
    df.drop(columns=['YYYYMMDD', 'HHMM'], inplace=True)
    df.rename(
        columns={
            'M(m/s)': 'WS',
            'D(deg)': 'WD',
            'T(C)': 'Temp',
            'De(k/m3)': 'Rho',
            'PRE(hPa)': 'Pressure',
            }, inplace=True
        )
    return df.astype(float)


def create_12x24(df, selected_col, output='column'):
    '''
    Create a 12x24 of a selected dataframe column

    Parameters
    ----------
    df : pd.DataFrame
        time series data you'd like to convert to a 12x24
    selected_col : undefined
        name of column you'd like to convert to a 12x24
        column must contain numeric data

    Returns
    -------
    df_12x24 : pd.DataFrame
        12x24 of selected column with columns=month and rows=hour

    '''
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    df_12x24 = pd.DataFrame(
        columns=range(1, 13),
        index=range(24))
    if output == '12x24':
        for month in range(1, 13):
            df_12x24[month] = [df[selected_col][(df['month'] == month) & (df['hour'] == hour)].mean() for hour in range(24)]
        df.drop(columns=['month', 'hour'], inplace=True)
        return df_12x24
    elif output == 'column':
        df['12x24'] = np.nan
        for month in range(1, 13):
            for hour in range(24):
                df['12x24'][(df['month'] == month) & (df['hour'] == hour)] = df[selected_col][(df['month'] == month) & (df['hour'] == hour)].mean()
        df.drop(columns=['month', 'hour'], inplace=True)
        return df
    else:
        raise ValueError('Unrecognized output type in create_12x24')


def fill_with_12x24(df, selected_col):
    """
    gap fill a column in a dataframe using data from a 12x24

    Parameters
    ----------
    df : pd.DataFrame
        data frame containing time series information. Index must be datetime objects.
    selected_col : str
        name of the column in df which you wish to gap fill using the 12x24 of the column.

    Returns
    -------
    df : pd.DataFrame
        input df object with gap filling in selected column.

    """
    # add a column made out of the 12x24
    df = create_12x24(df, selected_col)
    df[selected_col].fillna(df['12x24'], inplace=True)
    return df


def create_with_12x24(df_target, col_target, df_source, col_source):
    """
    create a column in a data frame using data from a 12x24. The 12x24 may be
    created from a different data frame (df_source)

    Parameters
    ----------
    df_target : pd.DataFrame
        data frame you wish to add a column to
    col_target : str
        name of the column you are adding to df_target
    df_source : pd.DataFrame
        data frame you wish to use to create the 12x24 for the target dataframe
    col_source : str
        name of the column you wish to convert to a 12x24 in the df_source data frame

    Returns
    -------
    df : pd.DataFrame
        input df object with one additional column, created from a 12x24.

    """
    # create a 12x24 from the source dataframe to populate the target dataframe
    df_12x24 = create_12x24(df_source, col_source, output='12x24')
    df_target[col_target] = np.nan
    df_target['month'] = df_target.index.month
    df_target['hour'] = df_target.index.hour
    for month in range(1, 13):
        for hour in range(24):
            df_target[col_target][(df_target['month'] == month) & (df_target['hour'] == hour)] = df_12x24[month][hour] * 1
    df_target.drop(columns=['month', 'hour'], inplace=True)
    return df_target


def append_values(df_target, col_target, df_source, col_source):
    """
    create a column in a data frame by copying time step from a second time series source. Note: the df_source
    data frame must contain data over the same time range of df_target, any additional time stamps in
    df_source will be ignored. If the df_source does not contain a time stamp within df_target, the function
    will fill all data points with the time stamp at the top of the hour.

    Parameters
    ----------
    df_target : pd.DataFrame
        data frame you wish to add a column to
    col_target : str
        name of the column you are adding to df_target
    df_source : pd.DataFrame
        data frame you wish to copy data from
    col_source : str
        name of the column you wish to copy data from

    Returns
    -------
    df : pd.DataFrame
        input df object with one additional column, created using the matching
        time stamps from df_source

    """
    # populate the target dataframe with the same time step from the source dataframe
    try:
        df_target[col_target] = [df_source[col_source][ind] for ind in df_target.index]
    except KeyError:
        # if you're populating 10 minute data from an hourly source
        df_target[col_target] = [df_source[col_source][ind.replace(minute=0)] for ind in df_target.index]
    return df_target


def calc_AirDensity(df, parameters):
    """
    calculate air density using temperature and pressure data

    Parameters
    ----------
    df : pd.DataFrame
        data frame representing time series data with columns containing temperature
        and air pressure data.
    parameters : dictionary
        dictionary containing relevant parameters for air density calculation.
        'TargetAirDensity' -> target air density for one specific hub height
        'HubHeightTargetAirDensity' -> hub height associated with target air density
        'PressureName' -> name of the column in df containing pressure data
        'TemperatureName' -> name of the column in df containing temperature data

    Returns
    -------
    df : pd.DataFrame
        Input data frame with additional column representing scaled air density data.

    """
    df['AirDensity'] = (
        df[parameters['PressureName'][1]] * 100) / (
            287 * (df[parameters['TemperatureName'][1]] + 273.15))
    fill_with_12x24(df, 'AirDensity') # fill gaps
    for height in parameters['DesiredHeights']: # scale to appropriate value
        scale_factor = (
            parameters['TargetAirDensity'] + (
                -0.0001181 * (height - parameters['HubHeightTargetAirDensity'])
                )
            ) / df['AirDensity'].mean()
        df[str(height) + '_rho'] = df['AirDensity'] * scale_factor
    df.drop(columns=['AirDensity'], inplace=True)
    return df


def calc_TI(df, parameters):
    """
    calculate time series turbulence intensity

    df : pd.DataFrame
        data frame representing time series data with columns containing wind speed
        and wind speed standard deviation data.
    parameters : dictionary
        dictionary containing relevant parameters for air density calculation.
        'DesiredHeight' -> hub height for which you want to calculate TI
        'HighSensorStd' -> name of column in df containing standard deviation data

    Returns
    -------
    df : pd.DataFrame
        Input data frame with additional column representing turbulence intensity data.

    """
    for height in parameters['DesiredHeights']:
        df[str(height) + '_TI'] = (
            df[parameters['HighSensorStd']] / df[str(height) + '_WS']
            ) * 100
        df = fill_with_12x24(df, str(height) + '_TI')
    return df


def shear_data(df, parameters, min_shear_percentile=0.01, max_shear_percentile=0.99):
    """
    extrapolate time series data to desired hub height using instantaneous shear values

    df : pd.DataFrame
        data frame representing time-series data with columns containing wind speed
        at two different heights.
    parameters : dictionary
        dictionary containing relevant parameters for air density calculation.
        'HighSensorName' -> name of column in df containing highest elevation wind
            speed measurements
        'HighSensorHeight' -> height of measurements represented by HighSensorName
        'LowSensorName' -> name of column in df containing lowest elevation wind
            speed measurements
        'LowSensorHeight' -> height of measurements represented by LowSensorName
    min_shear_percentile : float
        lower percentile threshold for identifying shear outliers. Shear values below this
        threshold will be overwritten with this threshold.
    max_shear_percentile : float
        upper percentile threshold for identifying shear outliers. Shear values above this
        threshold will be overwritten with this threshold.

    Returns
    -------
    df : pd.DataFrame
        Input data frame with additional column representing turbulence intensity data.

    """
    # create column of shear parameter
    df['shear'] = (
        np.log(df[parameters['HighSensorName']] / df[parameters['LowSensorName']])
        / np.log(parameters['HighSensorHeight'] / parameters['LowSensorHeight'])
        )
    # filter/cap shear outliers
    high_shear = df['shear'].quantile(max_shear_percentile)
    low_shear = df['shear'].quantile(min_shear_percentile)
    df['shear'][df['shear'] > high_shear] = high_shear
    df['shear'][df['shear'] < low_shear] = low_shear
    # shear data
    for height in parameters['DesiredHeights']:
        df[str(height) + '_WS'] = (
            df[parameters['HighSensorName']] * pow(
                height / parameters['HighSensorHeight'], df['shear'])
            )
    return df


def gap_fill_ws_wd(df, parameters):
    '''
    Gap fill wind speed and wind direction data using 12x24s made from the input time-series

    Parameters
    ----------
    df : pd.DataFrame
        data frame representing time-series data with columns containing wind speed and wind direction information
    parameters : dictionary
        dictionary containing relevant parameters for wind speed gap filling.
        'WindDirectionName' -> name of column in df containing wind direction measurements
        'DesiredHeights' -> list of desired heights of time-series wind speed data
            the data frame should already contain columns for each desired height with the column name <height>_WS

    Returns
    -------
    df : pd.DataFrame
        Input data frame with gap filling using the 12x24 of each wind speed and direction column

    '''
    df = fill_with_12x24(df, parameters['WindDirectionName'])
    for height in parameters['DesiredHeights']:
        df = fill_with_12x24(df, str(height) + '_WS')
    return df


def scale_WS_data(df, parameters):
    '''
    Scale the average wind speed of a time-series to the appropriate long-term value based on input
    long-term wind speeds at measurement height, global shear value, and desired extrapolated heights

    Parameters
    ----------
    df : pd.DataFrame
        data frame representing time-series data with columns containing wind speed information
    parameters : dictionary
        dictionary containing relevant parameters for wind speed gap filling.
        'DesiredHeights' -> list of desired heights of time series wind speed data
            the data frame should already contain columns for each desired height with the column name <height>_WS
        'LongTermWindSpeed' -> float of the long term wind speed at measurement height
        'ShearValue' -> float of global shear value at mast location. Note: current version does NOT allow for
            displacement heights

    Returns
    -------
    df : pd.DataFrame
        Input data frame with scaled wind speed columns to reflect desired long-term average wind speeds

    '''
    for height in parameters['DesiredHeights']:
        scaler = (
            parameters['LongTermWindSpeed']
            * pow(height / parameters['HighSensorHeight'],
                  parameters['ShearValue'])
            ) / df[str(height) + '_WS'].mean()
        df[str(height) + '_WS'] *= scaler
    return df


def create_OpenWindOuptut(df, parameters):
    '''
    Create an output *.csv file which can be loaded into OpenWind as a time-series mast object

    Parameters
    ----------
    df : pd.DataFrame
        data frame representing time-series data with columns containing wind speed, wind direction, temperature,
        air density, and turbulence intensity information
    parameters : dictionary
        dictionary containing relevant parameters for creating OpenWind met masts
        'DesiredHeights' -> list of desired heights of time series wind speed data
            the data frame should already contain wind speed, turbulence intensity, and air density columns for
            each desired height with the column names <height>_WS, <height>_TI, and <height>_rho respectively
        'OutputLocation' -> path to folder where time-series outputs will be saved

    Returns
    -------
    None

    '''
    # create additional columns
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    df['MOL'] = 0
    header_lines = 'MM, 1\n' + str(parameters['Longitude']) + ',' + str(parameters['Latitude']) + ','
    for height in parameters['DesiredHeights']:
        # create the CSV
        sub_df = df[[
            'Year',
            'Month',
            'Day',
            'Hour',
            'Minute',
            str(height) + '_WS',
            parameters['WindDirectionName'],
            parameters['TemperatureName'][1],
            str(height) + '_rho',
            str(height) + '_TI',
            'MOL'
            ]]
        sub_df.set_index('Year', inplace=True)
        file_name = os.path.join(
            parameters['OutputLocation'],
            str(parameters['MastName']) + '_' + str(height) + 'm.csv')
        sub_df.to_csv(file_name)
        mast_name = str(parameters['MastName']) + '_' + str(height) + 'm'
        header = header_lines + str(height) + '\n0,' + mast_name + ',' + mast_name + '\n'
        # add in header
        with open(file_name, 'r+') as file:
            content = file.readlines()
            content.insert(0, header)
            file.seek(0)
            content = "".join(content)
            file.write(content)
    df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace=True)
    pass


def create_masts(parameters):
    '''
    Wrapper for above functions, creating a long-term corrected and extrapolated time-series *.csv output
    in the format required by OpenWind

    Parameters
    ----------
    parameters : dictionary
        dictionary variable containing the following keys:
        'InputFiles': dictionary,
        'MastPrimarySource': unspecified type,
        'MastName': str,
        'Latitude': float,
        'Longitude': float,
        'TimeZone': int,
        'HighSensorName': str,
        'HighSensorHeight': float,
        'LongTermWindSpeed': float,
        'ShearValue': float,
        'HighSensorStd': str,
        'LowSensorName': str,
        'LowSensorHeight': float,
        'DesiredHeights': list of float like,
        'WindDirectionName': str,
        'TemperatureName': tuple (unspecified type, str),
        'PressureName': tuple (unspecified type, str),
        'TargetAirDensity': float,
        'HubHeightTargetAirDensity': float like,
        'gap_fill_WS_WD': bool,
        'OutputLocation': str

    Returns
    -------
    df : pd.DataFrame
        data frame containing time-series information for OpenWind mast creation

    '''
    # read in data
    for key, value in parameters['InputFiles'].items():
        # if data is windographer measurements
        if value[1].lower() in ['windographer', 'windog', 'measured', 'measurements', 'meas']:
            parameters['InputFiles'][key] = read_windog_file(value[0])
        # if data is vortex output
        elif 'vortex' in value[1].lower():
            parameters['InputFiles'][key] = read_vortex_file(value[0])
        # if data is from windographer data downloader
        elif value[1].lower() in ['era5', 'merra2']:
            parameters['InputFiles'][key] = read_windog_file(value[0], separator='\t')
            parameters['InputFiles'][key].index += pd.Timedelta(hours=parameters['TimeZone'])

    # set main dataframe to the correct input data
    df = parameters['InputFiles'][parameters['MastPrimarySource']]
    # fill Temp and pressure. Calculate, fill, and scale density columns
    if parameters['TemperatureName'][0] == parameters['MastPrimarySource']:
        # if you are using measured temperature data
        df = fill_with_12x24(df, parameters['TemperatureName'][1])
    elif parameters['TemperatureName'][1] == '12x24':
        # if you are using different temperature data and want to fill with the 12x24 values
        df = create_with_12x24(
            df,
            'Temperature',
            parameters['InputFiles'][parameters['TemperatureName'][0]],
            'Temp')
        parameters['TemperatureName'] = (parameters['TemperatureName'][0], 'Temperature')
    elif parameters['TemperatureName'][1] == 'TS':
        # if you are using different temperature data and want to fill with the TS values
        df = append_values(
            df,
            'Temperature',
            parameters['InputFiles'][parameters['TemperatureName'][0]],
            'Temp'
            )
        parameters['TemperatureName'] = (parameters['TemperatureName'][0], 'Temperature')

    if parameters['PressureName'][0] == parameters['MastPrimarySource']:
        # if you are using measured pressure data
        df = fill_with_12x24(df, parameters['PressureName'][1])
    elif parameters['PressureName'][1] == '12x24':
        # if you are using different pressure data and want to fill with the 12x24 values
        df = create_with_12x24(
            df,
            'Pressure',
            parameters['InputFiles'][parameters['PressureName'][0]],
            'Pressure')
        parameters['PressureName'] = (parameters['PressureName'][0], 'Pressure')
    elif parameters['PressureName'][1] == 'TS':
        # if you are using different pressure data and want to fill with the TS values
        df = append_values(
            df,
            'Pressure',
            parameters['InputFiles'][parameters['PressureName'][0]],
            'Pressure'
            )
        parameters['PressureName'] = (parameters['PressureName'][0], 'Pressure')

    df = calc_AirDensity(df, parameters)
    # shear wind speeds
    df = shear_data(df, parameters, limit_type='std')
    if parameters['gap_fill_WS_WD'] in ['12x24', True]:
        # gap fill WS and Dir with 12x24
        df = gap_fill_ws_wd(df, parameters)
    elif parameters['gap_fill_WS_WD'] not in [False, None]:
        raise ValueError("gap_fill_WS_WD not recognized. Please use '12x24' or None")
    df = calc_TI(df, parameters)
    df = scale_WS_data(df, parameters)
    create_OpenWindOuptut(df, parameters)
    return df


def determineDirectionSector(df, parameters, directionSectors=16):
    bin_width = 360. / directionSectors
    df['directionSector'] = ((df[parameters['WindDirectionName']] + (bin_width / 2)) / bin_width)
    df['directionSector'] = df['directionSector'].fillna(999)
    df['directionSector'] = df['directionSector'].astype(int)
    df['directionSector'][df['directionSector'] == directionSectors] = 0
    df['directionSector'] = df['directionSector'].replace(999, np.nan)
    return df


def format_tab(parameters, height, df, header):
    df.columns = [round(col * 100, 4) for col in df.columns]
    new_index = [i+0.5 for i in range(36)]
    df.index = new_index
    df = (df / df.sum() * 1000).round(3)
    filepath = os.path.join(parameters['OutputLocation'], parameters['MastName'] + '_' + str(height) + 'm.tab')
    df.to_csv(filepath, sep='\t')
    with open(filepath, 'r+') as file:
        content = file.readlines()
        content.insert(0, header)
        file.seek(0)
        content = "".join(content)
        file.write(content)
    return df


def create_tabs(parameters, df, threshold=1e-5):
    date = datetime.now()
    date = date.strftime('%Y/%m/%d')
    df = determineDirectionSector(df, parameters)
    for height in parameters['DesiredHeights']:
        target_ws = (
            parameters['LongTermWindSpeed']
            * pow(height / parameters['HighSensorHeight'], parameters['ShearValue']
                  )
        )
        header = (
            parameters['MastName'] + '_' + str(height) + 'm'
            + ' | ' + parameters['HighSensorName']
            + ' | ' + parameters['WindDirectionName']
            + ' | ' + 'Created via Python Script: ' + date + '\n'
            + '\t' + str(parameters['Latitude'])
            + ' ' + str(parameters['Longitude'])
            + ' ' + str(height) + '\n'
            + '\t\t16 1.00 0.00\n'
            )
        denom = len(df[(~df['directionSector'].isna()) & (~df[str(height) + '_WS'].isna())])
        cols = [len(df[(df['directionSector'] == i) & (~df[str(height) + '_WS'].isna())]) / denom for i in range(16)]
        index = [i for i in range(36)]
        index[0] = 0.25
        flag = True
        ct = 0
        df_tabs = []
        while flag:
            df_tab = pd.DataFrame(index=index)
            for ct, col in enumerate(cols):
                df_tab[col] = [len(df[(df[str(height) + '_WS'] > (i - 0.5)) & (df[str(height) + '_WS'] <= (i + 0.5)) & (df['directionSector'] == ct)]) for i in range(36)]
            ave_ws = np.dot(cols, [np.dot(df_tab.index, df_tab[col]) / df_tab[col].sum() for col in cols])
            if abs(ave_ws - target_ws) < threshold:
                flag = False
                df_tabs.append(df_tab)
                format_tab(parameters, height, df_tab, header)
            else:
                df[str(height) + '_WS'] *= target_ws / ave_ws
                ct += 1
            if ct == 50:
                raise ValueError('Taking too long to converge to target wind speed. Please confirm input data are correct.')
    return df_tabs


if __name__ == '__main__':
    my_params = {
        'InputFiles': {
            # label for source: (path to source, source format)
            0: (r'test_windographerReader_custom.txt', 'windographer'),
            1: (r'test_vortexReader.txt', 'vortex'),
            },
        'MastPrimarySource': 0,
        'MastName': 'MET001', # str
        'Latitude': 45., # float
        'Longitude': -120., # float
        'TimeZone': -8, # int
        'HighSensorName': 'Speed1', # str
        'HighSensorHeight': 60, # float
        'LongTermWindSpeed': 9.5, # float
        'ShearValue': 0.2, # float
        'HighSensorStd': 'Speed1SD', # str
        'LowSensorName': 'Speed2', # str
        'LowSensorHeight': 30, # float
        'DesiredHeights': [100], # list of floats
        'WindDirectionName': 'Direction',
        'TemperatureName': (1, '12x24'),
        'PressureName': (1, 'TS'),
        'TargetAirDensity': 1.225, # float
        'HubHeightTargetAirDensity': 100, # float like
        'gap_fill_WS_WD': True, # bool
        'OutputLocation': r'C:\Users\Annalise\Downloads' # str
        }
    df_TS = create_masts(my_params)
    create_tabs(my_params, df_TS)
    pass
