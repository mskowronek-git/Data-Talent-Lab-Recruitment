#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from tabulate import tabulate
from pymcdm import methods as mcdm_methods
from pymcdm import normalizations as norm
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Przetworzenie danych

# In[2]:


# importing data

dfKult = pd.read_csv('KULT_3881_CTAB_20220211231432.csv', sep=';')
dfLudn = pd.read_csv('LUDN_2425_CTAB_20220211224427.csv', sep=';')
dfTran = pd.read_csv('TRAN_3164_CTAB_20220211231704.csv', sep=';')
dfTury = pd.read_csv('TURY_3186_CTAB_20220211230207.csv', sep=';')
dfWyna = pd.read_csv('WYNA_2497_CTAB_20220211225034.csv', sep=';')

dfKult.fillna(0)
dfLudn.fillna(0)
dfTran.fillna(0)
dfTury.fillna(0)
dfWyna.fillna(0)


# In[3]:


# changing the names of the columns to be more readable

dfLudn = dfLudn.rename(columns = {'ludność na 1 km2;2020;[osoba]': 'Ludność na 1 km2'}, inplace = False)
dfTran = dfTran.rename(columns = {'ścieżki rowerowe (drogi dla rowerów) ogółem;2020;[km]': 'Ścieżki rowerowe'}, inplace = False)
dfTury = dfTury.rename(columns = {'ogółem;obiekty ogółem;2019;[ob.]': 'Turystyczne obiekty noclegowe'}, inplace = False)
dfWyna = dfWyna.rename(columns = {'ogółem;2020;[zł]': 'Przeciętne wynagrodzenie'}, inplace = False)

# convert commas to dots to enable format to float

dfTran = dfTran.stack().str.replace(',','.').unstack()
dfWyna = dfWyna.stack().str.replace(',','.').unstack()

dfWyna['Przeciętne wynagrodzenie'] = dfWyna['Przeciętne wynagrodzenie'].astype(float)
dfTran['Ścieżki rowerowe'] = dfTran['Ścieżki rowerowe'].astype(float)

dfWyna


# In[4]:


dfKult = dfKult.drop(labels=['Kod', 'Unnamed: 7'], axis='columns')
dfKult = dfKult.set_index('Nazwa')
dfKult


# In[5]:


# calculating the average number of sports facilities as a measure of the county's sports activity

dfKult['Średnia ilość obiektów'] = dfKult.mean(axis=1)
dfKult = dfKult.filter(['Średnia ilość obiektów'])
dfKult = dfKult.reset_index()
dfKult


# In[6]:


# combining a table with data into one

df_merged = pd.merge(dfKult, dfLudn, on=("Nazwa"), how="inner")  
df_merged = pd.merge(df_merged, dfTran, on=("Nazwa"), how="inner")  
df_merged = pd.merge(df_merged, dfWyna, on=("Nazwa"), how="inner")  
df_merged = pd.merge(df_merged, dfTury, on=("Nazwa"), how="inner")  

df_merged = df_merged.drop_duplicates(subset=['Nazwa']).reset_index(drop=True)


# In[7]:


df_merged = df_merged.drop(labels=['Kod_x', 'Unnamed: 3_x','Kod_y', 'Unnamed: 3_y'], axis='columns')
df_merged


# In[8]:


# creating a matrix that will be used for optimization

matrix = df_merged[df_merged.columns[1:]].to_numpy()
print(matrix)


# # Indeks: Wypożyczalnia rowerów

# In[9]:


df_merged.columns.tolist()


# In[10]:


# assigning weights to columns with values for optimization purposes
# value of weights assigned arbitrarily

weights = np.array([(1/10), (2/10), (3/10), (1/10), (3/10)])
print(weights)


# In[11]:


# determining the direction of the objective function for each of the columns, in this case all at max

types = np.array([1, 1, 1, 1, 1])


# In[12]:


# calculation of the matrix of results using the TOPSIS method

topsis = mcdm_methods.TOPSIS()
result = topsis(matrix, weights, types)
result


# In[13]:


# Convert index back to dataframe

indexWypozyczalnia = pd.DataFrame(result, columns=['Wartość indeksu']) 
indexWypozyczalnia['Powiat']= df_merged['Nazwa'] 
cols = indexWypozyczalnia.columns.tolist()
cols = cols[-1:] + cols[:-1]
indexWypozyczalnia = indexWypozyczalnia[cols]
indexWypozyczalnia


# In[14]:


column = indexWypozyczalnia['Wartość indeksu']
max = column.max()


# In[15]:


column = indexWypozyczalnia['Wartość indeksu']
min = column.min()


# In[16]:


# Performing a mathematical operation on the index, thanks to which the best result is 100 and the worst is 0

indexWypozyczalnia["Wartość indeksu"] = (((indexWypozyczalnia["Wartość indeksu"] - min)/(max - min))* 100).round(0).astype(int)
indexWypozyczalnia = indexWypozyczalnia.sort_values(by=["Wartość indeksu"], ascending=False)
indexWypozyczalnia = indexWypozyczalnia.reset_index()
indexWypozyczalnia = indexWypozyczalnia.drop(labels=['index'], axis='columns')


# In[17]:


#indexWypozyczalnia = indexWypozyczalnia.set_index(["Powiat"])
#indexWypozyczalnia.loc["Powiat oleski"]


# In[18]:


# creating a ranking of the best five counties

top5Wypozyczalnia = indexWypozyczalnia.head(5)
top5Wypozyczalnia.index += 1 
top5Wypozyczalnia = top5Wypozyczalnia.rename(columns = {'Powiat': 'Powiat - Wypożyczalnia rowerów', 'Wartość indeksu': 'Wartość indeksu - Wypożyczalnia rowerów'}, inplace = False)
top5Wypozyczalnia


# In[19]:


histWypozyczalnia = top5Wypozyczalnia.hist(bins=10, grid=False, figsize=(8,10), layout=(3,1), sharex=True, color='#86bf91', zorder=2, rwidth=0.9)
histWypozyczalnia


# # Indeks: Salon rowerowy

# In[20]:


df_merged.columns.tolist()


# In[21]:


# all the following operations are analogous to those for bike rentals

weights = np.array([(1.5/10), (2.5/10), (2.5/10), (3.5/10), (0/10)])
print(weights)


# In[22]:


types = np.array([1, 1, 1, 1, 1])


# In[23]:


topsis = mcdm_methods.TOPSIS()
result = topsis(matrix, weights, types)
result


# In[24]:


indexSalon = pd.DataFrame(result, columns=['Wartość indeksu']) 
indexSalon['Powiat']= df_merged['Nazwa'] 
cols = indexSalon.columns.tolist()
cols = cols[-1:] + cols[:-1]
indexSalon = indexSalon[cols]
indexSalon


# In[25]:


column = indexSalon['Wartość indeksu']
max = column.max()


# In[26]:


column = indexSalon['Wartość indeksu']
min = column.min()


# In[27]:


indexSalon["Wartość indeksu"] = (((indexSalon["Wartość indeksu"] - min)/(max - min))* 100).round(0).astype(int)
indexSalon = indexSalon.sort_values(by=["Wartość indeksu"], ascending=False)
indexSalon = indexSalon.reset_index()
indexSalon = indexSalon.drop(labels=['index'], axis='columns')


# In[28]:


top5Salon = indexSalon.head(5)
top5Salon.index += 1 
top5Salon = top5Salon.rename(columns = {'Powiat': 'Powiat - Salon rowerowy', 'Wartość indeksu': 'Wartość indeksu - Salon rowerowy'}, inplace = False)
top5Salon


# In[29]:


histSalon = top5Salon.hist(bins=10, grid=False, figsize=(8,10), layout=(3,1), sharex=True, color='#86bf91', zorder=2, rwidth=0.9)
histSalon


# # Zapis wyników do nowego pliku csv

# In[30]:


# Creating a common table for both indexes

top5Wypozyczalnia['Ranking'] = top5Wypozyczalnia.index
top5Salon['Ranking'] = top5Salon.index

dfResults = pd.merge(top5Wypozyczalnia, top5Salon, on='Ranking')
dfResults = dfResults.set_index('Ranking')
dfResults


# In[32]:


# final visualization of the table with indexes

cm = sns.light_palette("green", as_cmap=True)

th_props = [
  ('font-size', '11px'),
  ('text-align', 'center'),
  ('font-weight', 'bold'),
  ('color', '#6d6d6d'),
  ('background-color', '#f7f7f9')
  ]


td_props = [
  ('font-size', '11px'),
  ('font-weight', 'bold'),
  ('color', '#000000'),  
  ]

styles = [
  dict(selector="th", props=th_props),
  dict(selector="td", props=td_props)
  ]

(dfResults.style
  .background_gradient(cmap=cm, subset=['Wartość indeksu - Wypożyczalnia rowerów','Wartość indeksu - Salon rowerowy'])
  .highlight_max(subset=['Wartość indeksu - Wypożyczalnia rowerów','Wartość indeksu - Salon rowerowy'])
  .set_table_styles(styles))


# In[ ]:


dfResults.to_csv('wyniki.csv', sep='\t')

