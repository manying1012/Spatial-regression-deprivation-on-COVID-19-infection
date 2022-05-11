import pandas as pd
!pip install pysal
from pysal.lib import weights
from pysal.explore import esda
from pysal.model import spreg
from pysal.viz.splot.esda import moran_scatterplot, lisa_cluster, plot_local_autocorrelation
import geopandas as gpd
import numpy as np
import contextily as ctx
import pysal as ps

import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
import zipfile
import io

uploaded = files.upload()
zf = zipfile.ZipFile(io.BytesIO(uploaded['Local_Authority_Districts_(May_2021)_UK_BFE.zip']), "r")
zf.extractall()

LADnames_upload = files.upload()
LADnames = pd.read_csv(io.BytesIO(LADnames_upload["Total_covid_infections.csv"]))
LADnames = pd.read_csv(io.BytesIO(LADnames_upload["cleaned_covid_dataset (3).csv"]))
LADnames = pd.read_csv(io.BytesIO(LADnames_upload["20210716_cleaned_dataset_CRF.csv"]))
LADnames[' Infection per 100,000 '] = LADnames[' Infection per 100,000 '].str.strip().str.replace(',','').astype(float)
infections = LADnames[' Infection per 100,000 '].tolist()
age = LADnames['Median age'].tolist()
deprivation = LADnames['Proportion of LSAOs with most deprivated neighbourhoods'].tolist()
BAME = LADnames['Proportion of BAME'].tolist()
density = LADnames['People per sqm'].tolist()

shapefile = "Local_Authority_Districts_(May_2021)_UK_BFE.shp"
imd = gpd.read_file(shapefile)
imd = imd.set_index('LAD21CD', drop=False)
imd.head()
imd.info()

s = pd.Series(infections, index=LADnames['LAD19code'])
imd['infections'] = s
s2 = pd.Series(age, index=LADnames['LAD19code'])
imd['age'] = s2
s3 = pd.Series(np.log(density), index=LADnames['LAD19code'])
imd['density'] = s3
s4 = pd.Series(deprivation, index=LADnames['LAD19code'])
imd['deprivation'] = s4
s5 = pd.Series(np.log(BAME), index=LADnames['LAD19code'])
imd['BAME'] = s5
imd.info()
#print(imd.head())

imd = imd.drop('LAD21NMW', axis=1)
imd = imd.dropna()
imd.info()

#w = weights.KNN.from_dataframe(imd, k=1)
w = weights.Queen.from_dataframe(imd, idVariable='LAD21CD')
imd = imd.drop(w.islands)
w = weights.Queen.from_dataframe(imd, idVariable='LAD21CD')
w.transform = 'R'
#print('w', w.weights.values())
#print('w dic len', len(w.weights.values()))

#imd['w_infections'] = weights.lag_spatial(w, imd['infections'])
#imd.info()
#imd['infections_std'] = (imd['infections'] - imd['infections'].mean()) / imd['infections'].std()
#imd['w_infections_std'] = weights.lag_spatial(w, imd['infections_std'])

mi = esda.Moran(imd['infections'], w)
#mi = ps.Moran(imd['infections'], w)
print(mi.I)
print(mi.p_sim)

#print(imd['infections'])
#print(imd['age'])
#print(imd['density'])

YVar = imd[['infections']].values
#variables = ['age', 'density', 'BAME', 'deprivation']
variables = ['age', 'deprivation']
#XVars = [imd['age'].values, imd['density'].values]
XVars = variables

#imd = imd.to_numpy()
#YVar = imd[:,11]
#XVars = imd[:,[12,13]]

#print(YVar)
#print(YVar)

ols = spreg.OLS(imd[['infections']].values, imd[variables].values, w=w, name_y='infections', name_x=variables, nonspat_diag=False, spat_diag=True)
mi_regression = esda.Moran(ols.u, w, two_tailed=False)
pd.Series(index=['Morans I','Z-Score','P-Value'], data = [mi_regression.I, mi_regression.z_norm, mi_regression.p_norm])
#ols = spreg.OLS(YVar, XVars, w=w, name_y=YVar, name_x=XVars, nonspat_diag=False, spat_diag=True)
print(ols.summary)

lag = spreg.ML_Lag(imd[['infections']].values, imd[variables].values, w=w, name_y='infections', name_x=variables)
print(lag.summary)
mi = esda.Moran(lag.u, w, two_tailed=False)
pd.Series(index=['Morans I','Z-Score','P-Value'], data = [mi.I, mi.z_norm, mi.p_norm])
