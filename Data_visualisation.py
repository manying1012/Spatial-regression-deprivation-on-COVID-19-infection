import pandas as pd
import numpy as np
import io
from google.colab import files
from statistics import mean
from statistics import stdev

covid_infection = files.upload()
#covid_infection_code = files.upload()
covid_infection_dataframe = pd.read_csv(io.BytesIO(covid_infection['utla_2021-07-01.csv']))
#infection_LAD_code = pd.read_csv(io.BytesIO(covid_infection_code['infection_codes.csv']))
infection_LAD_code = pd.read_csv(io.BytesIO(covid_infection_code['2021_england LADs.csv']))

date = covid_infection_dataframe['date']
case = covid_infection_dataframe['newCasesByPublishDate']
area_code = covid_infection_dataframe['areaCode']
uinque_codes = infection_LAD_code['E_LADs']

LAD_rolling_avs = {}
rolling_av_date = []
total_infections = {}

for i in uinque_codes:
  indices = [area_code[j] and date[j] and case[j] for j, x in enumerate(area_code) if x == i]
  indices2 = [date[j] for j, x in enumerate(area_code) if x == i]
  indices3 = [j for j, x in enumerate(area_code) if x == i]

  # FINDING SUM OF TOTAL INFECTIONS FOR ENGLAND 
  #if len(indices2) >= 436:
  #  rolling_av_date = list(indices2[::7])
  #  rollingAverage = []
  #  step = 7
  #  for k, _ in enumerate(indices[::step]):
  #    sub_list = indices[k*7:] if (k+1)*7 > len(indices) else indices[k*7:(k+1)*7]  # Condition if the len(my_list) % step != 0
  #    rollingAverage.append(sum(sub_list)/float(len(sub_list)))
  #  LAD_rolling_avs[i] = rollingAverage
  #  total_infections_43[i] = sum(indices)

  rolling_av_date = list(indices2[::7])
  rollingAverage = []
  step = 7
  for k, _ in enumerate(indices[::step]):
    sub_list = indices[k*7:] if (k+1)*7 > len(indices) else indices[k*7:(k+1)*7]  # Condition if the len(my_list) % step != 0
    rollingAverage.append(sum(sub_list)/float(len(sub_list)))

  means = mean(rollingAverage)
  SD = stdev(rollingAverage)

  z_score = [(u - means)/SD for u in rollingAverage]
  LAD_rolling_avs[i] = z_score
  #LAD_rolling_avs[i] = rollingAverage
  #total_infections[i] = sum(indices)

rolling_av_date = rolling_av_date

#from google.colab import drive
#drive.mount('/content/drive')
#dataframe = pd.DataFrame(total_infections_43, index=[0])
#dataframe.to_csv("/content/drive/My Drive/Covid_project_dataset/revised_infections_dataset.csv", index=False, encoding='utf-8-sig')

# TEMPORAL HEATMAP
import seaborn as sns
import matplotlib.pyplot as plt

print(LAD_rolling_avs.keys())
LAD_rolling_avs = pd.DataFrame(LAD_rolling_avs)
f, ax = plt.subplots(figsize=(70, 30))
heatmap = sns.heatmap(LAD_rolling_avs, xticklabels = list(LAD_rolling_avs.keys()), yticklabels = rolling_av_date, linewidths=.05)
heatmap.figure.savefig("output.png")
files.download("output.png")
