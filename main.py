#Code author: Vivien Fisch-Romito, University of Geneva
#Conceptualization : Vivien Fisch-Romito, Evelina Trutnevyte, University of Geneva
#Date of modification: 14/12/23
#Description : Main script to launch in parallel different hindcasting runs


#import packages & library
import os
import numpy as np
import pandas as pd
import filepath  #module created
import load_curve #module created
import dispatch #module created
import capital_dynamic_3 #module created
import data_proc # module created
from IPython.display import clear_output
import sys
from multiprocessing import Pool



def run_country(country):
    
    try: 
    
        study_frame=pd.read_excel('study_frame.xlsx',
                            header=0)
        
        possible_tech=['Nuclear','HydroRoR','HydroDam','HardCoal','BrownCoal','Gas','Oil','Geothermal','OnshoreWind','OffshoreWind','PV',
                'Biomass','WasteIncineration','LandfillGas','Biogas','Storage','Import','Export','Storage_charging','Import/Export']

        input_path, output_path, plots_path = filepath.main(country)   

        #input data and reshaping dataframe
        
        elec_data=pd.read_excel(input_path + '/Input_data_' + country + '.xlsx',
                                header=[0],
                                index_col=[1,0],
                                sheet_name='Sheet1')
        elec_data=elec_data.stack().to_frame().reset_index()
        elec_data.columns = ['Year', 'Parameter', 'Techno', 'Value']
        elec_data=elec_data.pivot(index=['Year','Techno'],
                                columns='Parameter',
                                values='Value')

        country_data=pd.read_excel(input_path + '/Country_data_' + country + '.xlsx',
                                header=[0],
                                index_col=[0,1])
        country_data=country_data.reset_index().pivot(index='Year', columns='Parameter',values='Value')
        
    
        
        for s in range (0,64):    
            print('new run is ' + country + str(s))
            
            #Run conditions
            if (study_frame.loc[s,"ind_social"]==1) and (
                country  not in data_proc.soc_nuc.index.get_level_values(0).unique().values): 
                print('No data for social accceptance, the run will stop')
                continue
            
            if (study_frame.loc[s,"ind_gov"]==1) and (
                country  not in data_proc.oecd_gov.index.get_level_values(0).unique().values):
                print('No data for governance, the run will stop')
                continue
            
            if (study_frame.loc[s,"ind_finance"]==1) and (
            country  not in data_proc.wacc.index.get_level_values(0).unique().values):
                print('No data for investment risks, the run will stop')
                continue
                
            os.makedirs(output_path + '/Scenario/' + country + '/' + str(s), exist_ok=True)
            
            logfile = open(output_path + '/Scenario/' + country + '/' + str(s)+'/'+'console_output.txt', "w")
            stdout = sys.stdout
            sys.stdout = logfile
            
            
            #Initializing output dataframes
            try:
                gas_init=data_proc.ggpt.loc[country].copy()
            except:
                gas_init=pd.Series(data=1, index=[data_proc.average_year_g])
            try:    
                hcoal_init=data_proc.gcpt.loc[country].loc[:,'HardCoal'].copy()
            except: 
                hcoal_init=pd.Series(data=1, index=[data_proc.average_year_hc])
            try:
                nuke_init=data_proc.jrc_nuclear.loc[country].copy()
            except:
                nuke_init=pd.Series(data=1, index=[data_proc.average_year_nuke])
            
            
            bcoal_init=0
            if'BrownCoal' in elec_data.index.get_level_values(1).unique():
                try:
                    bcoal_init=data_proc.gcpt.loc[country].loc[:,'BrownCoal'].copy()
                except:    
                    bcoal_init=pd.Series(data=1, index=[data_proc.average_year_bc])
                
            
        
            output_data = pd.DataFrame(index=pd.MultiIndex.from_product([np.arange(1990, 2026),elec_data.index.get_level_values(1).unique().tolist()],names=["Year", "Techno"]))
            output_data=output_data.assign(cap_stock=0.0, cap_new=0.0, cap_closed=0.0, elec_gen=0.0, cap_new_inst=0.0)
            output_data.loc[1990,'cap_stock']=elec_data.loc[1990,'Actual_capacity'].values
            
            export_data=pd.DataFrame(index=range(1990,2020)).assign(Export=0.0)   #dataframe specific for exports
            charge_data=pd.DataFrame(index=range(1990,2020)).assign(Storage_charging=0.0) #dataframe specific for storage charge
            idx = pd.MultiIndex.from_product([np.arange(1990, 2020),possible_tech])
            hourly_output_data=pd.DataFrame(index=idx)
            
            
            #Transform capital age split in shares of 1990 stock (use total capital stock in 1990 as index). Only for Coal, Gas and Nuke for which we track the vintage from 1990
            hcoal_init.update(output_data.loc[(1990,'HardCoal'),'cap_stock']*hcoal_init/hcoal_init.sum())
            print('hcoal_init',hcoal_init)
            gas_init.update(output_data.loc[(1990,'Gas'),
                                            'cap_stock']*gas_init/gas_init.sum())
            nuke_init.update(output_data.loc[(1990,'Nuclear'),
                                            'cap_stock']*nuke_init/nuke_init.sum())
        
        
            if'BrownCoal' in elec_data.index.get_level_values(1).unique():
                bcoal_init.update(output_data.loc[(1990,'BrownCoal'),'cap_stock']*bcoal_init/bcoal_init.sum())
            
            #Start and end year of modelling
            y=1990
            ymax=2019
            
            while y<=ymax:
                print(y)
                print('load_curve')
                load_curve.main(y,country, country_data, elec_data)
                print('dispatch')
                dispatch.optimize(y, 
                                    country, 
                                    country_data, 
                                    elec_data, 
                                    output_data, 
                                    load_curve.Rep_days, 
                                    load_curve.size_cluster_ratio_annual, 
                                    load_curve.cluster_labels,
                                    export_data,
                                    charge_data, 
                                    hourly_output_data,
                                    load_curve.No_ed)

                print('capital_investments')
                if y<2019 : 
                    capital_dynamic_3.main(y, 
                                        country, 
                                        country_data, 
                                        elec_data,
                                        output_data, 
                                        load_curve.Proj_rep_days, 
                                        study_frame.loc[s],
                                        hcoal_init, 
                                        gas_init, 
                                        nuke_init, 
                                        bcoal_init,
                                        data_proc.soc_nuc, 
                                        data_proc.soc_wind,
                                        data_proc.soc_coal,
                                        data_proc.soc_gas,
                                        data_proc.soc_pv,
                                        data_proc.oecd_gov,
                                        data_proc.wacc
                                        )
                    
                y += 1


            sys.stdout = stdout
            logfile.close()
            
            output_data.to_csv(output_path + '/Scenario/' + country + '/' + str(s)+'/'+'output_data.csv')  
            charge_data.to_csv(output_path + '/Scenario/' + country + '/' + str(s)+'/'+'charge_data.csv')  
            export_data.to_csv(output_path + '/Scenario/' + country + '/' + str(s)+'/'+'export_data.csv')  
            #accuracy_indicators.to_csv(output_path + '/Scenario/' + country + '/' + str(s)+'/'+'accuracy_indicators.csv')  
            
            #os.system('clear')
            #clear_output(wait=False)

            
    #import cProfile
    #import re
    #cProfile.run('test()', sort= "cumulative" )


    except:
        print('error for ' , country,s)
        raise Exception ('Stop everything')
    
    
if __name__ == '__main__':

    countries=['LUX', 'LVA', 'MLT', 'NLD', 'NOR', 'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'SWE']
    #['AUT', 'BEL', 'BGR', 'CHE', 'CYP', 'CZE', 'DEU', 'DNK', 'ESP', 'EST', 'FIN', 'FRA',
     #          'GBR', 'GRC', 'HUN', 'HRV', 'IRL', 'ISL', 'ITA', 'LTU', 'LUX', 'LVA', 'MLT', 'NLD',
      #         'NOR', 'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'SWE']

    pool=Pool(processes=4)
    scenarios=pool.map_async(run_country, countries, chunksize=1)
    while not scenarios.ready():
        if not scenarios._success:
            print('Exiting for failure')
            pool.terminate()
            pool.join()
            break
