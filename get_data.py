'''
This file is provided by the HidroData repository from github.com/wallissoncarvalho/HidroData
Wallisson Moreira de Carvalho. (2020, April 16). HidroData: Working with Brazilian hydrological data (Version v1.0.1). Zenodo. http://doi.org/10.5281/zenodo.3755065
'''

import xml.etree.ElementTree as ET
import requests
import pandas as pd
import calendar
import geopandas as gpd
from tqdm import tqdm


def inventory(params={'codEstDE': '', 'codEstATE': '', 'tpEst': '1', 'nmEst': '', 'nmRio': '', 'codSubBacia': '',
                      'codBacia': '', 'nmMunicipio': '', 'nmEstado': '', 'sgResp': '', 'sgOper': '',
                      'telemetrica': ''}):
    """
    This function searches for stations registered in the Hidroweb inventory. By default, the code is looking for flow
    stations. To search for precipitation stations, change the tpEst argument to 2.

    To make a selection, you must create a dictionary with the following parameters:
        codEstDE: 8-digit station code - INITIAL (e.g., 00047000)
        codEstATE: 8-digit station code - FINAL (e.g., 90300000)
        tpEst: station type (i.e., 1- Precipitation or 2- Flow)
        nmEst: station name (e.g., Barra Mansa)
        nmRio: river name (e.g., Rio Javari)
        codSubBacia: sub-basin code (e.g., 10)
        codBacia: basin code (e.g., 1)
        nmMunicipio: municipality name (e.g., Itaperuna)
        nmEstado: state name (e.g., Rio de Janeiro)
        sgResp: acronym from the responsible for the station (e.g., ANA)
        sgOper: acronym from the responsible  for the station (e.g., CPRM)
        telemetrica: Is it a telemetric station? (i.e., 1-YES or 0-NO)

    To use this function, there is a standard dictionary with mandatory keys, and each value is optional:
    params = {'codEstDE': '', 'codEstATE': '', 'tpEst': '', 'nmEst': '', 'nmRio': '', 'codSubBacia': '', 'codBacia': '',
          'nmMunicipio': '', 'nmEstado': '', 'sgResp': '', 'sgOper': '', 'telemetrica': ''}
    """
    check_params = ['codEstDE', 'codEstATE', 'tpEst', 'nmEst', 'nmRio', 'codSubBacia',
                    'codBacia', 'nmMunicipio', 'nmEstado', 'sgResp', 'sgOper', 'telemetrica']
    if list(params.keys()) != check_params:
        print('You must pass the dictionary with the standard keys, use help(inventory) on the command line to '
              'understand the standard.')
        return

    response = requests.get('http://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroInventario', params)
    tree = ET.ElementTree(ET.fromstring(response.content))
    root = tree.getroot()
    if params['tpEst'] == '1':
        index = 1
        stations = pd.DataFrame(columns=['Name', 'Code', 'Type', 'DrainageArea', 'SubBasin', 'County', 'State',
                                         'Responsible', 'Latitude', 'Longitude'])
        for station in tqdm(root.iter('Table')):
            stations.at[index, 'Name'] = station.find('Nome').text
            code = station.find('Codigo').text
            stations.at[index, 'Code'] = f'{int(code):08}'
            stations.at[index, 'Type'] = station.find('TipoEstacao').text
            stations.at[index, 'DrainageArea'] = station.find('AreaDrenagem').text
            stations.at[index, 'SubBasin'] = station.find('SubBaciaCodigo').text
            stations.at[index, 'County'] = station.find('nmMunicipio').text
            stations.at[index, 'State'] = station.find('nmEstado').text
            stations.at[index, 'Responsible'] = station.find('ResponsavelSigla').text
            stations.at[index, 'Latitude'] = float(station.find('Latitude').text)
            stations.at[index, 'Longitude'] = float(station.find('Longitude').text)
            index += 1
    elif params['tpEst'] == '2':
        index = 1
        stations = pd.DataFrame(columns=['Name', 'Code', 'Type', 'SubBasin', 'County', 'State',
                                         'Responsible', 'Latitude', 'Longitude'])
        for station in tqdm(root.iter('Table')):
            stations.at[index, 'Name'] = station.find('Nome').text
            code = station.find('Codigo').text
            stations.at[index, 'Code'] = f'{int(code):08}'
            stations.at[index, 'Type'] = station.find('TipoEstacao').text
            stations.at[index, 'SubBasin'] = station.find('SubBaciaCodigo').text
            stations.at[index, 'County'] = station.find('nmMunicipio').text
            stations.at[index, 'State'] = station.find('nmEstado').text
            stations.at[index, 'Responsible'] = station.find('ResponsavelSigla').text
            stations.at[index, 'Latitude'] = float(station.find('Latitude').text)
            stations.at[index, 'Longitude'] = float(station.find('Longitude').text)
            index += 1
    else:
        print('Please choose a station type on the tpEst parameter.')
        return
    stations = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations.Longitude, stations.Latitude))
    return stations


def stations(list_station, data_type,only_consisted=False):
    """
    New
    Get the station data series from a list of stations code.
    :param list_station: list
    :param data_type: string, (e.g., '1' - for stage data, '2' - precipitation data,'3' - flow data)
    :param only_consisted, boolean, optional (if True return only consisted data), standard False
    :return: Pandas DataFrame
    """
    params = {'codEstacao': '', 'dataInicio': '', 'dataFim': '', 'tipoDados': '', 'nivelConsistencia': ''}
    data_types = {'3': ['Vazao{:02}'], '2': ['Chuva{:02}'], '1': ['Cota{:02}']}
    params['tipoDados'] = data_type
    data_stations = []

    for station in tqdm(list_station):
        params['codEstacao'] = str(station)
        try:
            response = requests.get('http://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroSerieHistorica', params,
                                    timeout=60.0)
        except (
                requests.ConnectTimeout, requests.HTTPError, requests.ReadTimeout, requests.Timeout,
                requests.ConnectionError):
            continue

        tree = ET.ElementTree(ET.fromstring(response.content))
        root = tree.getroot()
        df = []
        for month in root.iter('SerieHistorica'):
            code = month.find('EstacaoCodigo').text
            consist = int(month.find('NivelConsistencia').text)
            date = pd.to_datetime(month.find('DataHora').text, dayfirst=True)
            date = pd.Timestamp(date.year, date.month, 1, 0)
            last_day = calendar.monthrange(date.year, date.month)[1]
            month_dates = pd.date_range(date, periods=last_day, freq='D')
            data = []
            list_consist = []
            for i in range(last_day):
                value = data_types[params['tipoDados']][0].format(i + 1)
                try:
                    data.append(float(month.find(value).text))
                    list_consist.append(consist)
                except TypeError:
                    data.append(month.find(value).text)
                    list_consist.append(consist)
                except AttributeError:
                    data.append(None)
                    list_consist.append(consist)
            index_multi = list(zip(month_dates, list_consist))
            index_multi = pd.MultiIndex.from_tuples(index_multi, names=["Date", "Consistence"])
            df.append(pd.DataFrame({f'{int(code):08}': data}, index=index_multi))
        if (len(df))==0:
            continue
        df = pd.concat(df)
        df = df.sort_index()
        if only_consisted == False:
            drop_index = df.reset_index(level=1, drop=True).index.duplicated(keep='last')
            df = df[~drop_index]
            df = df.reset_index(level=1, drop=True)
        else:
            df = df[df.index.get_level_values(1) == 2]
            df = df.reset_index(level=1,drop=True)
            if (len(df)) == 0:
                continue
        series = df[f'{int(code):08}']
        date_index = pd.date_range(series.index[0], series.index[-1], freq='D')
        series = series.reindex(date_index)
        data_stations.append(series)
    data_stations = pd.concat(data_stations, axis=1)
    date_index = pd.date_range(data_stations.index[0], data_stations.index[-1], freq='D')
    data_stations = data_stations.reindex(date_index)
    return data_stations
