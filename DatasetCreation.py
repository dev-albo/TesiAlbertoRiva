# -*- coding: utf-8 -*-
"""
File principale di creazione del dataset a partire dal dataset disountOptionData.com
Sono necessari i file di supporto:
    - RiskFreePreprocessing.py: helper class per estrarre interpolazioni delle curve dei tassi risk-free US dollar
    - 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.dates as mdates

import pandas as pd
from sklearn import preprocessing
from IPython.utils import io

from scipy.stats import norm
from math import sqrt, exp, log, pi
from tqdm import tqdm
import time


#%%
"""Carico tutti i dati e applico le prime elaborazioni:


*   Elimino opzioni a scadenza 14 giorni
*   Actual365() sui giorni rimasti a scadenza
*   Elimino opzioni put
*   Call options predisposta per algoritmo di calcolo di implied volatility e pricatitioner's delta di Black-Scholes
*   Check del prezzo intrinseco (underlying price - strike price) ed eventuale scarto del record se prezzo intrinseco < 0
*   Estrazione di codice univoco per l'opzione
*   Salvataggio dei dati filtrati e elaborati


"""

data_path = r'C:\Users\Alberto\Documents\tesi\data\optionID.csv'
df = pd.read_csv(data_path, low_memory=(False))

initial_size = df.shape[0]
print('Initial size: {0}'.format(initial_size))

#%%

save = True
# save = False
save_path_base = r'C:\Users\Alberto\Documents\tesi\data\call_run\\'
path_filter_intrinsic = save_path_base + 'good_instrinsic.csv'
path_sorted = save_path_base + 'sorted.csv'
path_prices = save_path_base + 'clean_prices.csv'
risk_free_path = save_path_base +  'risk_free.csv'
risk_free_early_path = save_path_base +  'risk_free_early.csv'
delta_path = save_path_base + 'deltas.csv'
clean_delta_path = save_path_base + 'clean_deltas.csv'
lstm_path = save_path_base + 'lstm.csv'


#%%
df = df[df['ExpirationDate'] <= '2021-02-01'] 
df = df[df['DaysLeft'] >= 14]
df['DaysLeftActual'] = df['DaysLeft']/365
df['PutCall'] = df['PutCall'].apply(lambda optionType: 'c' if optionType == 'call' else 'p' )

#decido di tenere solo le opzioni call
df = df[df['PutCall'] == 'c']

# check prezzo implicito opzione
df['Intrinsic'] = df['UnderlyingPrice'] - df['StrikePrice'] - df['LastPrice']
df['correct'] = df['Intrinsic'].apply(lambda intr: 'good' if intr < 0 else 'bad')
df['optionkey'] = df['optionID'].apply(lambda real_name: real_name.split('.')[0])

good = df[df['correct'] == 'good']
good_size = good.shape[0]
print('Good size: {0}'.format(good_size))

# Salvo risultati dati filtrati
if save:
    good.to_csv(path_filter_intrinsic, index=False)

del good
del df

#%%

"""Drop di alcune colonne utilizzate come supporto per calcoli
    Calcolo numero opzioni da eliminare perche' prezzo sempre  = 0
"""

df = pd.read_csv(path_filter_intrinsic, low_memory=(False))

df_sort = df.sort_values(by = ['optionkey', 'DataDate'])
counts = df.groupby('optionkey')['LastPrice'].nunique()
zero_price = counts[counts != 1]

#STATS
print("Initial number of rows: {0}".format(initial_size))
print("Filtered number of rows: {0}".format(good_size))
print("Total number of option keys: {0}".format(len(counts)))
print("Options with price=0: {0}".format(len(zero_price)))
print("Data loss {0}".format(len(zero_price)/len(counts)))

# Drop useless columns
df_sort = df_sort.drop(['correct', 'Intrinsic', 'optionID'], axis=1)

if True:
    df_sort.to_csv(path_sorted , index=False)
    
del df
del df_sort


#%%


def calculateMultiProcessDeltaAndIV(ID, ns, rf_):
         """
        Calcolo della volatilita' implicita e della practitioner's Black-Sholes delta
        per ogni opzione.

        Il metodo viene chiamata in modo parallelo sul dataset, venendo cosi' eseguito
        parallelamente su piu' processi.


        Il metodo utilizza la libreria Mibian.

        Parametri:
            - ID: ID del processo che esegue la funzione
            - ns: multiprocessing.Manager: oggetto che permette di contenere porzione di dataset in modo threadSafe
                e di lanciare quindi questa funzione senza problemi di concorrenza sul dataset
            - rf_: curve dei tassi d'interesse risk-free

        Returns: 
            - Void: log su file dell'andamento del processo, i valori di ritorno si trovano
            all'interno del Manager (ns.ret = data)
        """
        path = r'C:\Users\Alberto\Documents\out\\' + str(ID) + ".txt"
        f = open(path, "w+")
        
        f.write('STARTED\n')
        f.flush()

        data = ns.df
        errors = []
        missing_dates = []
        start_time = time.time()
        delta = []
        
        for index, row in data.iterrows():
    
            v = 0
            try:
                # prendo curva risk-free da dizionario
                cubic = rf_[row['DataDate']]
            except KeyError:
                missing_dates.append((row['DataDate'], row['optionkey']))
                data.at[index,'delta'] = -100
                data.at[index,'iv'] =  -100
                data.at[index,'rf'] = -100
                continue
            
            try:
                # Calcolo tasso d'interesse risk-free dalla curva precedentemente estratta
                rf = cubic(row['DaysLeftActual'])
                data.at[index,'rf'] = rf
                
                # Calcolo volatilita' implicita a partire dall'opzione
                v = mibian.BS([row['UnderlyingPrice'], row['StrikePrice'], rf, row['DaysLeft']], callPrice=(row['LastPrice']))
                # Calcolo la practitioner's BS delta tramite la volatilita' implicita trovata al punto precedente
                d = mibian.BS([row['UnderlyingPrice'], row['StrikePrice'], rf, row['DaysLeft']], volatility=(v.impliedVolatility))
                # Aggiungo al dataset
                data.at[index,'delta'] = d.callDelta
                data.at[index,'iv'] =  v.impliedVolatility
                delta.append(d.callDelta)
            
            except Exception:
                errors.append((row['DataDate'], row['optionkey']))
                data.at[index,'delta'] = -100
                data.at[index,'iv'] =  -100
                data.at[index,'rf'] = -100
                
        ns.ret = data

        # Log running information
        delta_time = (time.time() - start_time)/60
        f.write("Thread {} finished\n Took: {} minutes".format(ID, delta_time))
        f.close()
        print("Thread {} finished".format(ID))
        
        return



#%%
"""Volatility and Delta calculation: (file: volatilityEval.py + riskFreePreprocessing.py (stored in drive))
  
*   Calcolo curve risk free con RiskFreePreprocessing.py
*   Calcolo di delta e implied volatility con libreria mibian, lanciando calcoli su 6 processi diversi contemporaneamente
"""

import mibian 
from importlib.machinery import SourceFileLoader
riskFreePreproc = SourceFileLoader("RiskFreePreprocessing", "./riskFreePreprocessing.py").load_module()

#%%
# Path al file con i tassi risk-free ed interpolazione Cubic per estrarre curve
risk_free = riskFreePreproc.RiskFreePreprocessing.getInterpolations(risk_free_path)
risk_free_early = riskFreePreproc.RiskFreePreprocessing.getInterpolationsEarly(risk_free_early_path)

# Merge delle due fonti dati su unico DataFrame
risk_free = {**risk_free, **risk_free_early}


#%%
# Import per calcolo parallelo su piu' processori
import threading
import multiprocessing
from multiprocessing import Process, Manager

df = pd.read_csv(path_sorted)

#SHAPE: 3661783
#%%

df['DataDate'] = pd.to_datetime(df['DataDate'])
procs = []

if __name__ == '__main__':
    
    print('Init main')
    
    mgr = Manager()
    
    jobs = []
    n_spaces = []
    i = 0
    t_count = 1 
    
    start = time.time()
    
    print(i)

    # Assegno a ogni processo 500.000 righe di dati da elaborare
    while i < 1532342:
        
        print('Init process: '+ str(t_count))
        
        k = i + 500000
        ns = mgr.Namespace()
        ns.df = df[i:k]
        n_spaces.append(ns)    
        proc = multiprocessing.Process(target=calculateMultiProcessDeltaAndIV, args=('T'+str(t_count), ns, risk_free.copy()))
        jobs.append(proc)
        i = k
        t_count = t_count + 1

    print(t_count)
        
    print('Starting processes')
    for j in jobs:
        j.start()
        
    print('Processes started') 
    for j in jobs:
        j.join()
            
    end = start - time.time()

    merged_results = pd.DataFrame()

    for name in n_spaces:
        merged_results = pd.concat([name.ret, merged_results])
    
    
    if save:
        merged_results.to_csv(delta_path, index=False)

#%%
"""Cleaning post-calcolo: 
    - Rimozione di opzioni con:
        - delta non  nel range desiderato, ovvero troppo deep in/out the money
        - vita dell'opzione troppo breve (inferiore a 10 giorni)
"""

data = pd.read_csv(delta_path)

def valid_delta(x):
    is_valid = (x is not None) and (( x > 0.05 and x < 0.95) or ( x > -0.95 and x < -0.05 ))
                     
    return is_valid

grouped = data.groupby('optionkey')

options = []
options_cnt = 0
options_to_remove = []
not_in_range = []
only_one_record = 0
good_options = []

for index, group in grouped:
    options.append(index)
    options_cnt = options_cnt + 1
    
    deltas = group['delta']
    
    if len(deltas) < 10:
        options_to_remove.append(index)
        only_one_record = only_one_record + 1
        continue
    
    delta_cnt = 0
    for delta in deltas:
        if not valid_delta(delta):
            delta_cnt = delta_cnt + 1
        if delta_cnt > 3:            
            not_in_range.append(index)
            options_to_remove.append(index)
            break
        

data_c  = data[~data['optionkey'].isin(options_to_remove)]
data_c = data_c[data_c['delta'].notna()]
data_final  = data_c[data_c['delta'] != -100]

if True:
    data_final.to_csv(clean_delta_path, index=False)


#%%

"""Calcolo variazione giornaliera S&P500 e append sul dataset   
    - Calcolo price_variation 
    - Drop delle di colonne non piu' utili

"""

data = pd.read_csv(clean_delta_path)
data = data.sort_values(by = ['DataDate'])
grouped = data.groupby('DataDate')
prices = []

df = pd.DataFrame(columns=["DataDate", "PriceChange"])

for index, group in grouped:
    prices.append((index, group['UnderlyingPrice'].unique()[0]))
    
i = 0
changes = []
changes_dict = {}
for date, price in prices:
    i = i + 1
    if(i == len(prices)):
        break
    
    change = (prices[i][1] - prices[i-1][1])/prices[i-1][1]
    df = df.append( { "DataDate": prices[i][0], "PriceChange": change}, ignore_index=True)
    changes.append((prices[i], change))
    changes_dict[prices[i][0]] = change

#%%

to_drop = ['ExpirationDate', 'PutCall', 'DaysLeft', 'rf', 'StrikePrice', 'LastPrice']
dropped = data.drop(to_drop, axis=1)

for index, row in dropped.iterrows():
    try:
        date_ = row['DataDate']
        dropped.at[index,'PriceChange'] = changes_dict[date_]
    except KeyError:
        dropped.at[index,'PriceChange'] = 0
        
dropped.DataDate = pd.to_datetime(dropped.DataDate)

if save:
    dropped.to_csv(path_prices , index=False)
    
    
#%%

#STATS

gg = dropped.groupby('optionkey')   

tot = 0
cnt = 0
lenghts = []

for index, row in gg:
    size = len(row['delta'])
    cnt = cnt + 1
    tot = tot + size
    lenghts.append(size)
    
mean = tot/cnt

print("Mean: {} - OptionsKeys: {} - Max: {} - Min: {}".format(mean, tot, np.max(lenghts), np.min(lenghts)))

#%%

"""Create dataset for LSTM: (files: RowContainer.py + lstm_preprocessing.py)
*   Create a structure like: ttm(t-1) delta(t-1) return_(t-1) ttm(t-2) delta(t-2) return_(t-2) IV(t)

*   un record del dataframe ha t0 , t1, t2 e come t3 il valore della vol implicita al tempo t4
"""
from importlib.machinery import SourceFileLoader
rowContainer = SourceFileLoader("RowContainer", "./RowContainer.py").load_module()

def GetRecord(group, start_index, end_index):
    
    record =  rowContainer.TimeData()
    
    for index in range(start_index, end_index):
        delta = group.at[index, 'delta']
        ttm = group.at[index, 'DaysLeftActual']
        return_ = group.at[index, 'PriceChange']
        key = group.at[index, 'optionkey']
        iv = group.at[index, 'iv']
        
        container = rowContainer.RowContainer(key, delta, return_, ttm, iv)
        record.AddDailyData(container)
        
    return record


def GetColumns(window_size):
    
    times = window_size - 1
    
    cols = []
    
    for time_ in range(times, 0, -1):
        base_append = " (t-" + str(time_) + ")"
        cols.append("key" + base_append)
        cols.append("delta" + base_append)
        cols.append("return" + base_append)
        cols.append("ttm" + base_append)
        cols.append("IV" + base_append)
    
    cols.append("IV (t)")
    
    return cols


def GetColumnsAtShift(shift):
    base_append = " (t-" + str(shift) + ")"
    k = "key" + base_append
    delta = "delta" + base_append
    ret = "return" + base_append
    ttm = "ttm" + base_append
    iv = "IV" + base_append
    
    return tuple((k, delta, ret, ttm, iv))

def ConvertToDf(records, window_size):
    
    df_columns = GetColumns(window_size)
    records_df = pd.DataFrame(columns=df_columns)   
    
    
    all_rows = []
    
    #https://www.kite.com/python/answers/how-to-fill-a-pandas-dataframe-row-by-row-in-python
    
    for record in records:
        
        back_days = window_size -1
        row = [] 
        
        for daily_data in record.Get():
            key, delta, return_, ttm, iv  = daily_data.GetValues()
            key_col, delta_col, return_col, ttm_col, iv_col = GetColumnsAtShift(back_days)
            
            if back_days == 0:
                row.append(iv)
                break
            
            row.append(key)
            row.append(delta)
            row.append(return_)
            row.append(ttm)
            row.append(iv)
            
            back_days =  back_days - 1
            
        
        all_rows.append(row)
    
    # un record del dataframe ha t0 , t1, t2 e come t3 il valore della vol implicita al tempo t4, in ogni
    # momento t ho 3 variabili: eg ttm(t-1) delta(t-1) return_(t-1) ttm(t-2) delta(t-2) return_(t-2) IV(t)
    #records_df.loc['column'] = pandas.Series({'optionkey':1, 'ttm':5, 'delta':2, 'return_':3})
    
    records_df = pd.DataFrame(all_rows, columns=df_columns)
    
    return records_df
    

def GetSeriesFromContainer(container):
    return

#%%    
from tqdm import tqdm
data = pd.read_csv(path_prices)

data = data.sort_values(by = ['optionkey', 'DataDate'])
grouped = data.groupby(['optionkey'])

ds = pd.DataFrame()

window = 2

transformed = []
days = []
records = []
cnt = 0

for index, group in grouped:
    number_eval_days = len(group)
    
    number_new_elements = number_eval_days - window + 1
    
    if(number_new_elements <= 0):
        continue
    
    #start_index = 0
    
    for start_index in range(number_new_elements):
        
        #print("Taking samples from {0} to {1}".format(start_index, start_index + window))
        records.append(GetRecord(group.reset_index(), start_index, start_index + window))
#%%

#funzione from Container to dataframe()
lstm_df = ConvertToDf(records, window)
if save:
    lstm_df.to_csv(lstm_path, index = False)

#%%
