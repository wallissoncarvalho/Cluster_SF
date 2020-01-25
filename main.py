'''
Created by the authors.
@wallissoncarvalho
@machadoyang
'''

import pandas as pd
import methods

observed_flows = pd.read_pickle(r'data\observed_flows.pkl')
natural_flows = pd.read_pickle(r'data\natural_flows.pkl')
natural_flows = natural_flows.loc[pd.to_datetime('01/01/1995'):pd.to_datetime('31/12/2015')]
reservoirs_discharge = pd.read_pickle(r'data\reservoirs_discharge.pkl')
reservoirs_discharge = reservoirs_discharge.loc[pd.to_datetime('01/01/1995'):pd.to_datetime('31/12/2015')]
affected_stations = pd.read_pickle(r'data\affected_stations.pkl')

observed_flows = methods.naturalize_flow(observed_flows,natural_flows,reservoirs_discharge,affected_stations)

