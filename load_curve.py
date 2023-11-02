#Code author: Vivien Fisch-Romito, University of Geneva
#Date of modification: 20/02/23
#Description: This module of the STONES model is used to build the load duration curves for the observed year and the expected one in the future

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from math import sqrt


def normalize(df):
    if df.to_numpy().max()==df.to_numpy().min():
        norm_df=df 
    else: 
        norm_df=(df-df.to_numpy().min())/(df.to_numpy().max()-df.to_numpy().min())
    return norm_df

def unnormalize(norm_df, df):
    result=norm_df*(df.to_numpy().max()-df.to_numpy().min())+df.to_numpy().min()
    return result

def error_gen(actual, rounded):
    divisor = sqrt(1.0 if actual < 1.0 else actual)
    return abs(rounded - actual) ** 2 / divisor

def round_to_sum(vector, total):
    n = len(vector)
    rounded = [int(x) for x in vector]
    up_count = total - sum(rounded)
    errors = [(error_gen(vector[i], rounded[i] + 1) - error_gen(vector[i], rounded[i]), i) for i in range(n)]
    rank = sorted(errors)
    for i in range(up_count):
        rounded[rank[i][1]] += 1
    return rounded

       
def main(y, country, country_data, elec_data) :         
    # Number of representative day 
    global No_ed
    No_ed = 6

    #Read the annual electrictiy demand profile
    try: 
        load_prof=pd.read_csv('./inputs/{0}/Profiles/{0}_{1}.csv'.format(country, y),index_col=0)
        load_prof=load_prof.fillna(method='bfill',axis=1)
        
    except:
        try:
            available_profiles = [f for f in os.listdir('./inputs/{0}/Profiles'.format(country)) if os.path.isfile(os.path.join('./inputs/{0}/Profiles'.format(country), f)) and country in f]
            first_profile=sorted(available_profiles)[0]
            load_prof=pd.read_csv('./inputs/{0}/Profiles/{1}'.format(country, first_profile),index_col=0)
            load_prof=load_prof.fillna(method='bfill',axis=1)
            
        except:
            load_prof=pd.read_csv('./inputs/_common/Datasets/GenericLoadProfile.csv', index_col=0)
            #load_prof=load_prof[[i for i in load_prof.columns if 'SensedHourly' in i]] 
            
    load_prof.index=pd.DatetimeIndex(load_prof.index)
    
    #Associate the renewables load factors data file of the year 
    country_codes=pd.read_excel('./inputs/_common/Datasets/ISO_CountryCodes.xlsx')
    country_codes.index=country_codes['Alpha-3 code']
    name=country_codes.loc[country,'Alpha-2 code']
    
    pv = pd.read_csv('./inputs/_common/Datasets/renewables_ninja/ninja_pv_country_'+name+'_merra-2_corrected.csv', header=2,index_col=0)
    wind = pd.read_csv('./inputs/_common/Datasets/renewables_ninja/ninja_wind_country_'+name+'_current-merra-2_corrected.csv', header=2,index_col=0)
    hydro_ror=pd.read_csv('./inputs/_common/Datasets/PECD_ROR_LF_daily.csv').set_index(['Alpha-3 code', 'date'])
    
    if country in hydro_ror.index.get_level_values(0).unique():
       
        hydro_ror=hydro_ror.loc[country, 'lf'].reset_index()   
       
    else:
        mean_hror_lf=(elec_data.loc[(y,'HydroRoR'),"LF_max"] + elec_data.loc[(y,'HydroRoR'),"LF_min"])/2
        hydro_ror=pd.DataFrame(data = {'date': hydro_ror.loc['FRA', 'lf'].index.get_level_values(0), 'lf': mean_hror_lf})

    hydro_ror=hydro_ror.sort_values('date')      
    hydro_ror['date'] = pd.to_datetime(hydro_ror.date)  
    hydro_ror=hydro_ror.append(pd.DataFrame({'date': pd.date_range(start=hydro_ror.date.iloc[-1], end='2019-12-31', freq='D', closed='right'),
                                            'lf': np.append(hydro_ror.lf.iloc[-365:],hydro_ror.lf.iloc[-365:])}))
    hydro_ror=hydro_ror.set_index('date')
    hydro_ror
    
    hydro_ror=pd.concat([hydro_ror]*24, axis=1)
    hydro_ror.columns=range(0,24)
    print('hydro_ror')
    
    
    if len(wind.columns)>1:
        no_wind_tech=2
        wind.rename({'offshore':'OffshoreWind','onshore':'OnshoreWind'},axis=1,inplace=True)
        wind.drop('national',axis=1,inplace=True)
    else:
        no_wind_tech=1
        wind.rename({'national':'OnshoreWind'},axis=1,inplace=True)
        wind['OffshoreWind']=0   
        
  
    wind.index=pd.DatetimeIndex(wind.index)

    wind_on=wind.assign(date=wind.index.date,
                        hour=wind.index.hour).reset_index().drop(['time',
                                                                  'OffshoreWind'],1).pivot(index='date',
                                                                                        columns='hour',
                                                                                        values='OnshoreWind')   
    wind_off=wind.assign(date=wind.index.date,
                        hour=wind.index.hour).reset_index().drop(['time',
                                                                  'OnshoreWind'],1).pivot(index='date',
                                                                                        columns='hour',
                                                                                          values='OffshoreWind')
    
    pv.index=pd.DatetimeIndex(pv.index)
    pv.columns=['PV']
    pv=pv.assign(date=pv.index.date, hour=pv.index.hour).reset_index().drop('time',1).pivot(index='date',
                                                                                            columns='hour',
                                                                                            values='PV')

    
    hydro_ror.index=pd.DatetimeIndex(hydro_ror.index)
    
    
    #Creation of a matrix with normalized inputs for representative days
    if no_wind_tech==2:      
        X=normalize(load_prof).join(normalize(pv), lsuffix='a', rsuffix='b').join(normalize(hydro_ror), lsuffix='c', rsuffix='d').join(normalize(wind_on),lsuffix='e', rsuffix='f').join(normalize(wind_off),lsuffix='g', rsuffix='h')
        
    else : 
        X=normalize(load_prof).join(normalize(pv), lsuffix='a', rsuffix='b').join(normalize(hydro_ror), lsuffix='c', rsuffix='d').join(normalize(wind_on),lsuffix='e', rsuffix='f')
        
        #.join(wind_off,lsuffix='x', rsuffix='y')
    print(X)
    X.drop((X[(X.index.month==2) & (X.index.day==29)]).index, inplace = True)
    
    #Clustering to obtain n representatives days (with different weights)
    n_clusters=No_ed
    kmeansmodel = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    global cluster_labels
    cluster_labels = kmeansmodel.fit_predict(X)
    
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    size_cluster = np.zeros(n_clusters)
    
    
    # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster[i]= ith_cluster_silhouette_values.shape[0]
              
    global size_cluster_ratio_annual
    size_cluster_ratio_annual = size_cluster/sum(size_cluster)
    print('size_cluster ', size_cluster)
    
    
    #Extraction of cluster center to get representative days
    rep_days_data=kmeansmodel.cluster_centers_
    
    #Changing the shape of matrix, to get one time series per line and not normalized anymore
    global Rep_days
    if no_wind_tech==2: 
        Rep_days=pd.DataFrame(index=['HourlyDem', 'PV','HydroRoR', 'OnshoreWind', 'OffshoreWind'],
                    data=np.array([unnormalize(rep_days_data[:,0:24].flatten(), load_prof)/1000,
                                  unnormalize(rep_days_data[:,24:48].flatten(), pv), 
                                  unnormalize(rep_days_data[:,48:72].flatten(), hydro_ror), 
                                  unnormalize(rep_days_data[:,72:96].flatten(), wind_on),
                                  unnormalize(rep_days_data[:,96:120].flatten(), wind_off)]))
    else : 
        Rep_days=pd.DataFrame(index=['HourlyDem', 'PV','HydroRoR', 'OnshoreWind', 'OffshoreWind'],
                    data=np.array([unnormalize(rep_days_data[:,0:24].flatten(), load_prof)/1000,
                                  unnormalize(rep_days_data[:,24:48].flatten(), pv), 
                                  unnormalize(rep_days_data[:,48:72].flatten(), hydro_ror), 
                                  unnormalize(rep_days_data[:,72:96].flatten(), wind_on),
                                  unnormalize(rep_days_data[:,72:96].flatten(), wind_on)]))
    
    
    
    #Scaling hourly demand values proportionnally to the cumulative annual demand
    ElSupplied_s=np.zeros(n_clusters)
    for i in range(n_clusters):
            ElSupplied_s[i]=sum(Rep_days.loc['HourlyDem'].values[i*24:(i+1)*24])*365*size_cluster_ratio_annual[i]/1000
    
    Eldemand=country_data.loc[y,'ElSupplied_annual_central'] \
                  -elec_data.loc[(y,'Storage'),'Actual_generation'] \
                  -country_data.loc[y,'Distribution_losses']
    
    Rep_days.loc['HourlyDem']=Rep_days.loc['HourlyDem']*Eldemand/sum(ElSupplied_s)
    Rep_days[Rep_days < 0] = 0 #Because of the Kmeans usage, some outputs are slightly below 0, so modelling patch here
    print('Rep_days', Rep_days)   
    
    
    #Projection of annual hourly demand in five years (for capacity evolution)
    global Proj_rep_days
    Proj_rep_days=Rep_days.copy()
    
    #Replacement of peak values with the maximum known in the past + reserves
    #Proj_rep_days.loc['HourlyDem']=np.where(Proj_rep_days.loc['HourlyDem']==np.max(Proj_rep_days.loc['HourlyDem']), (1+country_data.loc[y,'Capacity_margin'])*np.mean(country_data.loc[y-5:y,'PeakDem_fromzero_central']),Proj_rep_days.loc['HourlyDem'])
    
    #Projected growth : average growth rate value based on the last 5 years
    
    if y<1991:
        Mean_dem=np.mean(country_data.loc[y-5:y,'PeakDem_fromzero_central'].values)
        Proj_peak=Mean_dem*1.02**5
        
        
    else:
        n=max(len(country_data.loc[y-5:y,'ElSupplied_annual_central'].values)-1,1)
        mean_annual_growth_rate=(country_data.loc[y-5:y,'ElSupplied_annual_central'].values[-1]/country_data.loc[y-5:y,'ElSupplied_annual_central'].values[0])**(1/n)-1
        print('mean annual growth', mean_annual_growth_rate) 
        Proj_peak=max(country_data.loc[y,'PeakDem_fromzero_central']*(1+mean_annual_growth_rate)**5,
                  np.max(country_data.loc[1990:y,'PeakDem_fromzero_central'].values))
        #Proj_peak=country_data.loc[y,'PeakDem_fromzero_central']+Proj_growth*5
        
    print('Projected peak electricity demand in five years', Proj_peak)
    
    
    
    #Scaling representative days proportionnaly to peak demand value    
    Proj_rep_days.loc['HourlyDem']=1*Proj_peak/np.max(Proj_rep_days.loc['HourlyDem'])*Proj_rep_days.loc['HourlyDem']
 
    #Repeat each represenative days by the number of days they represent, to obtain hourly value for the entire year  
    for i in range(n_clusters):
        Proj_rep_days=pd.concat([Proj_rep_days,
                  pd.concat([Proj_rep_days.iloc[:,i*24:(i+1)*24]]*(round_to_sum(size_cluster_ratio_annual*365, 365)[i]-1), axis=1, ignore_index=True)],
                  axis=1,ignore_index=True)
    
    #Sorting value to obtain a annual load duration curve (LDC) while keeping consistent renewables load factor values associated
    Proj_rep_days=Proj_rep_days.sort_values(by='HourlyDem', axis=1, ascending=False)


   
    
    