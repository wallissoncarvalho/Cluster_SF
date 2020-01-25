'''
Created by the authors.
@wallissoncarvalho
@machadoyang
'''

import pandas as pd
import methods

observed_flows = pd.read_pickle(r'data\observed_flows.pkl')
natural_flows = pd.read_pickle(r'data\natural_flows.pkl')
reservoirs_discharge = pd.read_pickle(r'data\reservoirs_discharge.pkl')
affected_stations = pd.read_pickle(r'data\affected_stations.pkl')

observed_flows = methods.naturalize_flow(observed_flows,natural_flows,reservoirs_discharge,affected_stations)

