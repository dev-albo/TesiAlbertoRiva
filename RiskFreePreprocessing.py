# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:43:20 2021

@author: alberto.riva
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import collections


#%%

class RiskFreePreprocessing:
    """
    Utility class to load and interpolate risk-free rates on riskless USD
    I suoi metodi estraggono una curva interpolata con metodo CubicSpline a partire da due file .csv
    contenenti i tassi a 1M, 3M, 6M, 1Y, 2Y per due diversi periodi di tempo.

    I metodi ritornano un dizionario <Data, coefficienti di interpolazione>
    """
    
    
    @staticmethod
    def getInterpolations(data_path):
        """
        Get dictionary with <Date>:<interpolation_coefficients> 
        Parameters
        ----------
        data_path : string
            Csv with risk-free rates

        Returns
        -------
        Dictionary

        """
        yields_df = pd.read_csv(data_path)
        
        yields_df['DateTime'] = pd.to_datetime(yields_df['Date'], format='%d/%m/%Y')
        
        grouped_by_date = yields_df.groupby(['DateTime'])

        interpolations_data = {}
        
        for index, group in grouped_by_date:
            ttm = RiskFreePreprocessing.convertToNumberOfDays(group['Point'])
            values = group['Value'].tolist()
            interpolations_data[index] = (ttm, values)
            
        
        interpolations = {}
        for key in interpolations_data:
            points = interpolations_data[key][0]
            values = interpolations_data[key][1]
            
            cubic_spline = interpolate.CubicSpline(points, values)
            interpolations[key] = cubic_spline
            
            #xs = np.arange(0.0, 5, 0.1)
            #fig, ax = plt.subplots(figsize=(6.5, 4))
            #ax.plot(points, values, 'o', label='data')
            #ax.plot(xs, cubic_spline(xs), label="S")
            
            #plt.savefig(fname=r'C:\Users\alberto.riva\Documents\tesi\plot.png')
        
        return collections.OrderedDict(sorted(interpolations.items()))
    
    @staticmethod
    def getInterpolationsEarly(data_path):
        
        yields_df = pd.read_csv(data_path)

        
        yields_df['DateTime'] = pd.to_datetime(yields_df['Date'], format='%d/%m/%Y')

        points_ = ['1M', '3M', '6M', '1Y', '2Y']

        interpolations_data = {}
        
        for index, row in yields_df.iterrows():
            ttm = RiskFreePreprocessing.convertToNumberOfDays(points_)
            values = []
            #values = group['Value'].tolist()
            
            values.append(row['1M'])
            values.append(row['3M'])
            values.append(row['6M'])
            values.append(row['1Y'])
            values.append(row['2Y'])
            
            interpolations_data[row['DateTime']] = (ttm, values)
                    
                
            interpolations = {}
            for key in interpolations_data:
                    
                    points = interpolations_data[key][0]
                    values = interpolations_data[key][1]
                    
                    cubic_spline = interpolate.CubicSpline(points, values)
                    interpolations[key] = cubic_spline
                    
                    #xs = np.arange(0.0, 5, 0.1)
                    #fig, ax = plt.subplots(figsize=(6.5, 4))
                    #ax.plot(points, values, 'o', label='data')
                    #ax.plot(xs, cubic_spline(xs), label="S")
                    
                    #plt.savefig(fname=r'C:\Users\alberto.riva\Documents\tesi\plot.png')
                
        return collections.OrderedDict(sorted(interpolations.items()))    
    
        
    
    
    @staticmethod
    def convertToNumberOfDays(points):
        """
        From strings like "1M" to 30 days to actualized days 30/365.

        Parameters
        ----------
        points : TYPE
            Points are string that need to be converted into an actual number of days

        Returns
        -------
        List of actualized days

        """
        ttm = []
        for point in points:
            
            if len(point) == 2:
                times = int(point[0])
                period = point[1]
            else:
                times = int(point[:2])
                period = point[2]
            
            period_days = 0
            if period == 'M':
                period_days = 30
            else:
                period_days = 365
                
            ttl = (times*period_days)/365
            ttm.append(ttl)
        
        ttm.sort()
        return ttm