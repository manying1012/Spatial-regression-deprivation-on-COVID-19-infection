import pandas as pd
import io
from google.colab import files
LADnames_upload = files.upload()
Infections_upload = files.upload()
Deaths_upload = files.upload()
Population_age_upload = files.upload()
Deprivation_upload = files.upload()
Deprivation_codes_upload = files.upload()

LADnames = pd.read_csv(io.BytesIO(LADnames_upload['Local_Authority_Districts_(December_2019)_Boundaries_UK_BFC.csv']))
Infections = pd.read_csv(io.BytesIO(Infections_upload['Total_covid_infections.csv']))
Deaths = pd.read_csv(io.BytesIO(Deaths_upload['Total_covid_deaths.csv']))
Population_age = pd.read_csv(io.BytesIO(Population_age_upload['Population age.csv']))
Deprivation = pd.read_csv(io.BytesIO(Deprivation_upload['Deprivation_data.csv']))
Deprivation_unique_codes = pd.read_csv(io.BytesIO(Deprivation_codes_upload['LAD_codes_deprivation.csv']))

data = {}
list_of_LAD_codes = LADnames['lad19cd'].tolist()
list_of_LAD_names = LADnames['lad19nm'].tolist()

# MERGE INFECTION DATA
infections_list = ['n/a']*382 
infection_LAD = Infections['Row Labels'].tolist()
infection_LAD_name = Infections['Area name'].tolist()
infection_numbers = Infections['Sum of newCasesByPublishDate'].tolist()

#print(len(infection_numbers))
#print(len(infection_LAD))
#print(len(infections_list))
#print(len(list_of_LAD_codes))

missing_infections = []
for i in infection_LAD:
  #print(i)
  if i not in list_of_LAD_codes:
    list_of_LAD_codes.append(i)
    list_of_LAD_names.append(infection_LAD_name[infection_LAD.index(i)])
    missing_infections.append(i)
  else:
    index = list_of_LAD_codes.index(i)
    #print(list_of_LAD_codes[index])
    #print(index)
    #print(infections_list[index])
    #print(infection_numbers[infection_LAD.index(i)])
    infections_list[index] = infection_numbers[infection_LAD.index(i)]
    #print(infections_list)

for x in missing_infections:
  index = infection_LAD.index(x)
  infections_list.append(infection_numbers[index])

#print(missing)
#print(list_of_LAD_codes)
#print(infections_list)

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
    #print(death_numbers[death_LAD.index(i)])
    #print(index)
    death_list[index] = death_numbers[death_LAD.index(i)]

# print(death_list)
#for x in missing_deaths:
#  index = death_LAD.index(x)
#  death_list.append(death_numbers[index])


# MERGE POPULATION AGE DATA 
no_LADS = len(list_of_LAD_codes)
population_density_list = ['n/a']*no_LADS
age_list = ['n/a']*no_LADS
pop_LAD = Population_age['LAD code'].tolist()
pop_LAD_names = Population_age['Area name'].tolist()
pop_density_numbers = Population_age['People per sqm'].tolist()
age_numbers = Population_age['Median age'].tolist()

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

#print(missing_pop)
for x in missing_pop:
  index = pop_LAD.index(x)
  population_density_list.append(pop_density_numbers[index])
  age_list.append(age_numbers[index])

no_add_LADS = len(list_of_LAD_codes) - no_LADS
add_list = ['n/a']*no_add_LADS
infections_list.extend(add_list)
death_list.extend(add_list)

#print(len(list_of_LAD_codes))
#print(len(list_of_LAD_names))
#print(len(infections_list))
#print(len(death_list))
#print(len(population_density_list))
#print(len(age_list))

# CLEAN SOCIAL DEPRVATION DATA 
no_LADS = len(list_of_LAD_codes)
social_deprivation = ['n/a']*no_LADS
deprivation_decile = Deprivation_upload['Index of Multiple Deprivation (IMD) Decile'].tolist()
dep_LAD_codes = Deprivation_upload['Local Authority District code (2019)'].tolist()
dep_LAD_names = Deprivation_codes_upload['Name_formatted'].tolist()
dep_LAD_unique_codes = Deprivation_codes_upload['LAD Code'].tolist()


missing_pop = []
for i in dep_LAD_unique_codes:
  if i not in list_of_LAD_codes:
    list_of_LAD_codes.append(i)
    list_of_LAD_names.append(dep_LAD_names[dep_LAD_unique_codes.index(i)])
    missing_pop.append(i)
  else:
    subdf = Deprivation[Deprivation.loc['Local Authority District code (2019)'] == i]
    Decile = subdf['Index of Multiple Deprivation (IMD) Decile'].tolist()
    print(subdf)
    total_LSAO = dep_LAD_codes.count(i)
    numb_1decile = Decile.count(1)

# FORM DATAFRAME
data['LAD19code'] = list_of_LAD_codes
data['LAD19name'] = list_of_LAD_names
data['Infections'] = infections_list
data['Deaths'] = death_list
data['People per sqm'] = population_density_list
data['Median age'] = age_list

#from google.colab import drive
#drive.mount('/content/drive')
#dataframe = pd.DataFrame(data)
#dataframe.to_csv("/content/drive/My Drive/Covid_project_dataset/cleaned_covid_dataset.csv", index=False, encoding='utf-8-sig')
