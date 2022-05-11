import pandas as pd
import numpy as np
import io
from google.colab import files
#covid_dataset = files.upload()
#covid_dataframe = pd.read_csv(io.BytesIO(covid_dataset['experimental_infection_dataset.csv']))
#covid_dataframe = pd.read_csv(io.BytesIO(covid_dataset['20210709_cleaned_dataset_CRF.csv']))
covid_dataframe = pd.read_csv(io.BytesIO(covid_dataset['20210716_cleaned_dataset_CRF.csv']))
covid_dataframe = covid_dataframe.fillna(covid_dataframe.mean())
#print(covid_dataframe)
#print(covid_dataframe.columns)


# VARIABLE SET UP ##############################################################
  # dependent variables 
#infections = np.log(covid_dataframe['Infections'])
death_rate = covid_dataframe[' Death per 100,000 ']
infection_rate =  covid_dataframe[' Infection per 100,000 '].str.strip().str.replace(',','').astype(float)
Case_fatality_rate = covid_dataframe['Case fatality rate']

  #independent variables 
density = covid_dataframe['People per sqm']
age = covid_dataframe['Median age']
BAME = covid_dataframe['Proportion of BAME']
deprivation = covid_dataframe['Proportion of LSAOs with most deprivated neighbourhoods']
#deprivation_transform = [np.arcsin(i) for i in deprivation]
#print(deprivation_transform)
#deprivation = [1 + i for i in covid_dataframe['Proportion of LSAOs with most deprivated neighbourhoods']]
#print(deprivation)
#diabetes = covid_dataframe['diabetes QOF']
#CHD = covid_dataframe['CHD QOF']
#hypertension = covid_dataframe['hypertension QOF']

x = covid_dataframe[['People per sqm', 'Median age', 'Proportion of BAME', 'Proportion of LSAOs with most deprivated neighbourhoods']]
print(x.mean())
print(x.std())

print(infection_rate.mean())
print(infection_rate.std())
#Y = Case_fatality_rate
#print(x)
#print(Y)


# Finding correlations and multicollinearity ###################################
pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.DataFrame(x)
corrM = df.corr()
#print(corrM)

import matplotlib as plt
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation of Variables')
plt.show()

  # VIF score 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
X = add_constant(df)
VIF = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
print(VIF)

  # R2 score - have to be the same scale??????
corr_matrix = np.corrcoef(deprivation, BAME)
corr = corr_matrix[0,1]
R_sq = corr**2
#print('r2 score for deprivation', R_sq)

corr_matrix = np.corrcoef(age, BAME)
corr = corr_matrix[0,1]
R_sq = corr**2
#print('r2 score for age on bame', r2)


# Descriptive statistics ######################################################
import statistics 

list1 = [density, BAME, age, deprivation, Case_fatality_rate, death_rate, infection_rate]
list_names = ['density', 'BAME', 'age', 'deprivation', 'Case_fatality_rate', 'death_rate', 'infection_rate']

for i in range(0, len(list1)):
  mean = round(statistics.mean(list1[i]),2)
  SD = round(statistics.stdev(list1[i]),2)
  print(list_names[i], ' mean, SD:', mean, SD)

# VISUALISATION ##############################################################
from scipy import stats
sns.regplot(x = deprivation, y = infection_rate)
#plt.scatter(deprivation, Case_fatality_rate)
#plt.show()

  #transorming data
density = np.log(density)
BAME = np.log(BAME)
Case_fatality_rate = np.log(Case_fatality_rate)
#deprivation = stats.boxcox(deprivation, lmbda = 1)
#deprivation, fitted_lambda = stats.boxcox(deprivation)


#list1 = [density, BAME, age, deprivation, CHD, Case_fatality_rate, death_rate, infection_rate]
#list_names = ['density', 'BAME', 'age', 'deprivation', 'CHD', 'Case_fatality_rate', 'death_rate', 'infection_rate']

#for i in range(0, len(list1)):
  #num_bins = 9
  #n, bins, patches = plt.hist(list1[i], num_bins, facecolor='blue', alpha=0.5)
  #plt.xlabel(list_names[i])
  #plt.ylabel('Frequency')
  #plt.show()


# Building a better model with Backward Elimination ###########################
import statsmodels.api as sm
diction = {}
#diction['density'] = density
diction['age'] = age
#diction['BAME'] = BAME
diction['deprivation'] = deprivation

#transformed_dataset = covid_dataframe[['People per sqm', 'Median age', 'Proportion of BAME', 'Proportion of LSAOs with most deprivated neighbourhoods']]
#transformed_dataset = covid_dataframe[['Median age', 'Proportion of LSAOs with most deprivated neighbourhoods']]

Y = infection_rate
#Y = death_rate
#Y = Case_fatality_rate
#transformed_dataset = np.append(arr = np.ones((123,1)).astype(int), values = transformed_dataset, axis = 1)
transformed_dataset = pd.DataFrame(diction)
transformed_dataset = sm.add_constant(transformed_dataset)

#x_opt = np.array(transformed_dataset[:, [0,1,2,3,4]], dtype=float)
#regressor_ols = sm.OLS(endog = Y, exog = x_opt).fit() 

regressor_ols = sm.OLS(endog = Y, exog = transformed_dataset).fit() 
print(regressor_ols.summary())
#print(x_opt)


# LINEAR REGRESSION MODEL ####################################################
#from sklearn.model_selection import train_test_split
#x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.2)
#print(deprivation)
#print(infection_rate)

#deprivation = deprivation.reshape(-1, 1)
from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
#regressor.fit(deprivation, infection_rate)

#Y_pred = regressor.predict(deprivation)
#r2 = regressor.score(deprivation, infection_rate)
#print(regressor.coef_)
#print(r2)

from statsmodels.api import OLS
#model = OLS(infection_rate,deprivation).fit().summary()
#print(model)

import matplotlib as mpl
#mpl.pyplot.scatter(Y_pred, Y_test)
