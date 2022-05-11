import pandas as pd
import io
from google.colab import files
LADnames_upload = files.upload()
Infections_upload = files.upload()
Deaths_upload = files.upload()
Population_age_upload = files.upload()
Deprivation_upload = files.upload()
Deprivation_codes_upload = files.upload()
Ethnicity_codes = files.upload()
hypertensionupload = files.upload()
CHDupload = files.upload()
diabetesupload = files.upload()

LADnames = pd.read_csv(io.BytesIO(LADnames_upload['Local_Authority_Districts_(December_2019)_Boundaries_UK_BFC.csv']))
Infections = pd.read_csv(io.BytesIO(Infections_upload['Total_covid_infections.csv']))
Deaths = pd.read_csv(io.BytesIO(Deaths_upload['Total_covid_deaths.csv']))
Population_age = pd.read_csv(io.BytesIO(Population_age_upload['Population age.csv']))
Deprivation = pd.read_csv(io.BytesIO(Deprivation_upload['Deprivation_data.csv']))
Deprivation_unique_codes = pd.read_csv(io.BytesIO(Deprivation_codes_upload['LAD_codes_deprivation.csv']))
Ethnicity = pd.read_csv(io.BytesIO(Ethnicity_codes['ethnicity.csv']))
hypertension = pd.read_csv(io.BytesIO(hypertensionupload['hypertension.csv']))
CHD = pd.read_csv(io.BytesIO(CHDupload['CHD QOF prevalence LAD.csv']))
diabetes = pd.read_csv(io.BytesIO(diabetesupload['20210707_diabetes_LAD.csv']))

data = {}
list_of_LAD_codes = LADnames['lad19cd'].tolist()
list_of_LAD_names = LADnames['lad19nm'].tolist()

# MERGE INFECTION DATA
infections_list = ['n/a']*382 
infection_LAD = Infections['Row Labels'].tolist()
infection_LAD_name = Infections['Area name'].tolist()
infection_numbers = Infections['Sum of newCasesByPublishDate'].tolist()

missing_infections = []
for i in infection_LAD:
  #print(i)
  if i not in list_of_LAD_codes:
    list_of_LAD_codes.append(i)
    list_of_LAD_names.append(infection_LAD_name[infection_LAD.index(i)])
    missing_infections.append(i)
  else:
    index = list_of_LAD_codes.index(i)
    infections_list[index] = infection_numbers[infection_LAD.index(i)]

for x in missing_infections:
  index = infection_LAD.index(x)
  infections_list.append(infection_numbers[index])

# MERGE DEATH DATA
no_LADS = len(list_of_LAD_codes)
death_list = ['n/a']*no_LADS
death_LAD = Deaths['Row Labels'].tolist()
death_numbers = Deaths['Sum of newOnsDeathsByRegistrationDate'].tolist()

missing_deaths = []
for i in death_LAD:
  if i not in list_of_LAD_codes:
    list_of_LAD_codes.append(i)
    missing_deaths.append(i)
  else:
    index = list_of_LAD_codes.index(i)
    death_list[index] = death_numbers[death_LAD.index(i)]

# MERGE POPULATION AGE DATA 
no_LADS = len(list_of_LAD_codes)
population_density_list = ['n/a']*no_LADS
age_list = ['n/a']*no_LADS
population_size = ['n/a']*no_LADS

pop_LAD = Population_age['LAD code'].tolist()
pop_LAD_names = Population_age['Area name'].tolist()
pop_density_numbers = Population_age['People per sqm'].tolist()
age_numbers = Population_age['Median age'].tolist()
population_numbers = Population_age['Estimated population'].tolist()

missing_pop = []
for i in pop_LAD:
  if i not in list_of_LAD_codes:
    list_of_LAD_codes.append(i)
    list_of_LAD_names.append(pop_LAD_names[pop_LAD.index(i)])
    missing_pop.append(i)
  else:
    index = list_of_LAD_codes.index(i)
    population_density_list[index] = pop_density_numbers[pop_LAD.index(i)]
    age_list[index] = age_numbers[pop_LAD.index(i)]
    population_size[index] = population_numbers[pop_LAD.index(i)]

for x in missing_pop:
  index = pop_LAD.index(x)
  population_density_list.append(pop_density_numbers[index])
  age_list.append(age_numbers[index])
  population_size.append(population_numbers[pop_LAD.index(i)])


no_add_LADS = len(list_of_LAD_codes) - no_LADS
add_list = ['n/a']*no_add_LADS
infections_list.extend(add_list)
death_list.extend(add_list)

# CLEAN SOCIAL DEPRVATION DATA 
no_LADS = len(list_of_LAD_codes)
social_deprivation = ['n/a']*no_LADS
deprivation_decile = Deprivation['Index of Multiple Deprivation (IMD) Decile'].tolist()
dep_LAD_codes = Deprivation['Local Authority District code (2019)'].tolist()
dep_LAD_names = Deprivation_unique_codes['Name_formatted'].tolist()
dep_LAD_unique_codes = Deprivation_unique_codes['LAD Code'].tolist()

missing_dep = []
for i in dep_LAD_unique_codes:
  if i not in list_of_LAD_codes:
    list_of_LAD_codes.append(i)
    list_of_LAD_names.append(dep_LAD_names[dep_LAD_unique_codes.index(i)])
    missing_dep.append(i)
  else:
    subdf = Deprivation[Deprivation['Local Authority District code (2019)'] == i]
    Decile = subdf['Index of Multiple Deprivation (IMD) Decile'].tolist()
    total_LSAO = dep_LAD_codes.count(i)
    numb_1decile = Decile.count(1)
    dep_propor = round(numb_1decile/total_LSAO,3)
    
    index = list_of_LAD_codes.index(i)
    social_deprivation[index] = dep_propor


# MERGING ETHNICITY
no_LADS = len(list_of_LAD_codes)
BAME_propor = ['n/a']*no_LADS
eth_code = Ethnicity['LAD code'].tolist()
eth_name = Ethnicity['LAD names'].tolist()
eth_pro = Ethnicity['Proportion of non-BAME'].tolist()

missing_eth = []
for i in eth_code:
  if i not in list_of_LAD_codes:
    list_of_LAD_codes.append(i)
    list_of_LAD_names.append(eth_name[eth_code.index(i)])
    missing_eth.append(i)
  else:
    index = list_of_LAD_codes.index(i)
    BAME_propor[index] = eth_pro[eth_code.index(i)]

no_add_LADS = len(list_of_LAD_codes) - no_LADS
add_list = ['n/a']*no_add_LADS
infections_list.extend(add_list)
death_list.extend(add_list)
population_density_list.extend(add_list)
age_list.extend(add_list)
social_deprivation.extend(add_list)
population_size.extend(add_list)


for x in missing_eth:
  index = eth_code.index(x)
  BAME_propor.append(eth_pro[index])


# CHD
no_LADS = len(list_of_LAD_codes)
CHD_propor = ['n/a']*no_LADS
chd_area_Code = CHD['Area Code'].tolist()
chd_name = CHD['AreaName'].tolist()
chd_pro = CHD['Value'].tolist()

missing_chd = []
for i in chd_area_Code:
  if i not in list_of_LAD_codes:
    missing_chd.append(i)
  else:
    index = list_of_LAD_codes.index(i)
    CHD_propor[index] = chd_pro[chd_area_Code.index(i)]

# DIABETES
no_LADS = len(list_of_LAD_codes)
db_propor = ['n/a']*no_LADS
db_area_Code = diabetes['Category Type'].tolist()
db_name = diabetes['Category'].tolist()
db_pro = diabetes['Count'].tolist()

missing_db = []
for i in db_area_Code:
  if i not in list_of_LAD_codes:
    missing_db.append(i)
  else:
    index = list_of_LAD_codes.index(i)
    db_propor[index] = db_pro[db_area_Code.index(i)]

# HYPERTENSION
no_LADS = len(list_of_LAD_codes)
hpt_propor = ['n/a']*no_LADS
hpt_area_Code = hypertension['Area Code'].tolist()
hpt_name = hypertension['AreaName'].tolist()
hpt_pro = hypertension['Value'].tolist()

missing_hpt = []
for i in hpt_area_Code:
  if i not in list_of_LAD_codes:
    missing_hpt.append(i)
  else:
    index = list_of_LAD_codes.index(i)
    hpt_propor[index] = hpt_pro[hpt_area_Code.index(i)]

#print(len(list_of_LAD_codes))
#print(len(list_of_LAD_names))
#print(len(infections_list))
#print(len(death_list))
#print(len(population_density_list))
#print(len(age_list))
#print(len(social_deprivation))
#print(len(BAME_propor))
#print(len(db_propor))
#print(len(CHD_propor))
#print(len(hpt_propor))


# DEPENDENT VARIABLES  
Infection_per_100000 = []
for x,y in zip(infections_list, population_size):
  if x != 'n/a' and y != 'n/a':
    Infection_per_100000.append(100000*(float(x) / float(y)))
  else:
    Infection_per_100000.append('n/a')

Death_per_100000 = []
for x,y in zip(death_list, population_size):
  if x != 'n/a' and y != 'n/a':
    Death_per_100000.append(100000*(float(x) / float(y)))
  else:
    Death_per_100000.append('n/a')

Case_fatality_rate = []
for x,y in zip(death_list, infections_list):
  if x != 'n/a' and y != 'n/a':
    Case_fatality_rate.append(float(x) / float(y))
  else:
    Case_fatality_rate.append('n/a')

# FORM DATAFRAME
data['LAD19code'] = list_of_LAD_codes
data['LAD19name'] = list_of_LAD_names
data['Infections'] = infections_list
data['Deaths'] = death_list
data['Infection per 100,000'] = Infection_per_100000
data['Death per 100,000'] = Death_per_100000
data['Case fatality rate'] = Case_fatality_rate
data['Population estimate'] = population_size
data['People per sqm'] = population_density_list
data['Median age'] = age_list
data['Proportion of LSAOs with most deprivated neighbourhoods'] = social_deprivation
data['Proportion of BAME'] = BAME_propor
data['CHD QOF'] = CHD_propor
data['diabetes QOF'] = db_propor
data['hypertension QOF'] = hpt_propor

#from google.colab import drive
#drive.mount('/content/drive')
dataframe = pd.DataFrame(data)
dataframe.to_csv("/content/drive/My Drive/Covid_project_dataset/cleaned_covid_dataset.csv", index=False, encoding='utf-8-sig')
