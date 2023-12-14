#Code author: Vivien Fisch-Romito, University of Geneva
#Conceptualization : Vivien Fisch-Romito, Evelina Trutnevyte, University of Geneva
#Date of modification: 14/12/23
#

import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.util.infeasible import log_infeasible_constraints


def replace_na(a,b):
    if pd.isna(a):
        r=b
    else:
        r=a
    return r

def optimize(y, country, country_data, elec_data, output_data, rep_days, size_cluster_ratio_annual, cluster_labels, export_data, charge_data, hourly_output_data, No_ed) :  
           
    #Correction to avoid hourly demand value below observed base demand and above observed peak, because of the clustering approach and the load profiles
    
    
    rep_days.loc['HourlyDem',rep_days.loc['HourlyDem']<=country_data.loc[y,'BaseDem_central']]=country_data.loc[y,'BaseDem_central']
    rep_days.loc['HourlyDem',rep_days.loc['HourlyDem']>=country_data.loc[y,'PeakDem_fromzero_central']]=country_data.loc[y,'PeakDem_fromzero_central']
    
    print('min hourly dem ', min(rep_days.loc['HourlyDem']))
    print('base dem central', country_data.loc[y,'BaseDem_central'])
    print('max hourly dem', max(rep_days.loc['HourlyDem']))
    print('peak dem central', country_data.loc[y,'PeakDem_fromzero_central'])
    
    
    output_data_y=output_data.loc[y]
    country_data_y=country_data.loc[y]
    elec_data_y=elec_data.loc[y]
    

    ### Initialize model optimization instance
    model = ConcreteModel() 
    
    #Snapshots with all representative days aggregated
    Snapshots_h = np.linspace(1,No_ed*24,No_ed*24).astype(int)
    
    #Snapshots with representative days differentiated
    Snapshots_h_s=np.zeros((No_ed,24))  
    
    #Junctions between representative days
    Junctions=np.zeros(No_ed)
    for i in range(No_ed):
        Junctions[i]=i*24+1
        Snapshots_h_s[i,:]=np.linspace(i*24+1,(i+1)*24,24)

    tech=elec_data.index.get_level_values(1).unique()
    tech=tech.drop(['Import','Storage'])
    
    ### define decision variables
    # Generator power output
    model.p = Var(tech, Snapshots_h, domain = NonNegativeReals)
    
    # Import / Export power output
    model.imp_p = Var(['Import'],Snapshots_h, domain = NonNegativeReals)
    model.exp_p = Var(['Export'],Snapshots_h, domain = NonNegativeReals)
    
    # Pseudo-line power flow
    model.line_p = Var(['Line'], Snapshots_h, domain = Reals)
    
    #Storage flows
    model.stor_d = Var(['Storage'],Snapshots_h, domain = NonNegativeReals)
    model.stor_c = Var(['Storage_charging'],Snapshots_h, domain = NonNegativeReals)
    
    #Storage intra representative days state of charge 
    model.soc_intra=Var(['SOC'],Snapshots_h, domain = Reals)
    
    #Storage inter representative days state of charge 
    model.soc_inter=Var(['SOC'],np.arange(1,len(cluster_labels)+1), domain = NonNegativeReals)
    
    ## Setting constraints
    # for every flexible generator p(t) < p_nom * LF_max  and p(t) > p_nom * LF_min
    model.p_cons = ConstraintList()
    for c in tech :
        for t in Snapshots_h :
            if c in ['PV','HydroRoR','OnshoreWind','OffshoreWind']:
                if ((rep_days.loc[c].values[0]==rep_days.loc[c].values).all()):
                    
                    model.p_cons.add(expr = model.p[c,t] <= output_data_y.loc[c,'cap_stock']*elec_data_y.loc[c,"LF_max"]*1.1)
                    model.p_cons.add(expr = model.p[c,t] >= output_data_y.loc[c,'cap_stock']*elec_data_y.loc[c,"LF_min"])
                    
                else:
                    model.p_cons.add(expr = model.p[c,t] <=output_data_y.loc[c,'cap_stock'] * max(rep_days.loc[c].values[t-1],0))
                    model.p_cons.add(expr = model.p[c,t] >=output_data_y.loc[c,'cap_stock'] *0.9*max(rep_days.loc[c].values[t-1],0))                            
            else :
                model.p_cons.add(expr = model.p[c,t] <= output_data_y.loc[c,'cap_stock']*min(elec_data_y.loc[c,"LF_max"]*1.1,1))
                
    #Adding more flexibility to be consistent with data            
            if c in ['Nuclear']: 
                model.p_cons.add(expr = model.p[c,t] >= output_data_y.loc[c,'cap_stock']*0.4)
            if c in ['BrownCoal', 'HardCoal']:
                model.p_cons.add(expr = model.p[c,t] >= output_data_y.loc[c,'cap_stock']*0.3)
            if c in ['HydroDam']:
                model.p_cons.add(expr = model.p[c,t] >= output_data_y.loc[c,'cap_stock']*elec_data_y.loc[c,"LF_min"])
            
    
    
    
    # Total supplied electricity = Domestic NEP (net electricity production) + net imports
    #Find domestic net electricity production
    Domestic_NEP=country_data.loc[:,'ElSupplied_annual_central'].values-elec_data.loc[pd.IndexSlice[:,'Import'],'Actual_generation'].values 
    
    #Maximum historical share of absolute net exchanges in domestic production
    Max_historical_exchange=np.abs(elec_data.loc[pd.IndexSlice[:,'Import'],'Actual_generation'].values/Domestic_NEP).max()
    print('Max_historical_exchange',Max_historical_exchange)
  

    
    
    #import power constraint
    model.i_cons = ConstraintList()
    for t in Snapshots_h:
        model.i_cons.add(expr = model.imp_p['Import',t] <= output_data_y.loc['Import','cap_stock'])
    # export power constraint
    model.e_cons = ConstraintList()
    for t in Snapshots_h:
        model.e_cons.add(expr = model.exp_p['Export',t] <= output_data_y.loc['Import','cap_stock'])   
        
    # Import - export - Pseudo-line flow = 0
    # create constraint list for nodal power balances
    model.n2 = ConstraintList()
    for t in Snapshots_h:
        n2_expr = 0
        n2_expr = model.imp_p['Import',t] - model.exp_p['Export',t] - model.line_p['Line',t] 
        model.n2.add(expr=n2_expr==0)  

        model.n2.add(expr= model.imp_p['Import',t] + model.exp_p['Export',t] <= output_data_y.loc['Import','cap_stock'])   
    
    
    #storage charge power constraint
    model.stor_c_cons = ConstraintList()
    for t in Snapshots_h:
        model.stor_c_cons.add(expr = model.stor_c['Storage_charging',t] <= output_data_y.loc['Storage','cap_stock'])
    # storage discharge power constraint
    model.stor_d_cons = ConstraintList()
    for t in Snapshots_h:
        model.stor_d_cons.add(expr = model.stor_d['Storage',t] <= output_data_y.loc['Storage','cap_stock'])    

    if output_data_y.loc['Storage','cap_stock']>0:
        #state of charge intra representative days
        model.soc_intra_cons=ConstraintList()
        for t in [x for x in Snapshots_h[1:] if x not in Junctions]:
            model.soc_intra_cons.add(expr=model.soc_intra['SOC',t]==
                               model.soc_intra['SOC',t-1]+model.stor_c['Storage_charging',t-1]*0.9-model.stor_d['Storage',t-1]/0.9)

        model.soc_intra_initial_cons=ConstraintList()
        for t in Junctions:
            model.soc_intra_initial_cons.add(expr=model.soc_intra['SOC',t]==0)


        #state of charge inter representative days
        model.soc_inter_cons=ConstraintList()
        for d in np.arange(2,len(cluster_labels)+1):
            k=cluster_labels[d-2] #representative day associated with the day d-1
            model.soc_inter_cons.add(expr=model.soc_inter['SOC',d]==model.soc_inter['SOC',d-1]+
                                     model.soc_intra['SOC', (k+1)*24]+model.stor_c['Storage_charging',(k+1)*24]*0.9-model.stor_d['Storage',(k+1)*24]/0.9)

        #closing loop constraint for soc inter days
        model.soc_inter_initial_cons = Constraint(expr=model.soc_inter['SOC',1]==model.soc_inter['SOC',len(cluster_labels)]+
                                     model.soc_intra['SOC', (cluster_labels[-1]+1)*24]+model.stor_c['Storage_charging',(cluster_labels[-1]+1)*24]*0.9-model.stor_d['Storage',(cluster_labels[-1]+1)*24]/0.9)

        #maximum state of charge inter days (maximum capacity of storage capacity)
        max_storage_cap=pd.read_csv('./inputs/_common/Datasets/PHS_storage_cap.csv', index_col=0).loc[:,"Energy Storage Capacity"].squeeze()
        model.soc_inter_max_cons=ConstraintList()
        model.soc_inter_min_cons=ConstraintList()
        for d in np.arange(1,len(cluster_labels)+1):
            k=cluster_labels[d-1] #representative day associated with the day d
            for g in range(1,25):
                model.soc_inter_max_cons.add(expr=model.soc_inter['SOC',d]+model.soc_intra['SOC', k*24+g]
                                             <=max_storage_cap.loc[country])
                model.soc_inter_min_cons.add(expr=model.soc_inter['SOC',d]+model.soc_intra['SOC', k*24+g]
                                             >=0)
            
            
            

    # balance between supply and demand          
    model.dem_cons = ConstraintList()
    dem_bal_expr = 0
                     
    for t in Snapshots_h:
        dem_bal_expr =  sum(model.p[c,t]*(1-elec_data_y.loc[c,"Own_use"]) for c in tech)+model.line_p['Line',t]*(1-elec_data_y.loc['Import',"Own_use"])-model.stor_c['Storage_charging',t]+model.stor_d['Storage',t]
        model.dem_cons.add(expr= dem_bal_expr==rep_days.loc['HourlyDem'].values[t-1]*(1)) #to add the transmission losses
                     
    # power ramp rate constraint
    model.p_ramp_cons = ConstraintList()
    for c in tech:
            for t in [x for x in Snapshots_h[1:] if x not in Junctions]:
                model.p_ramp_cons.add(expr =  model.p[c,t] - model.p[c,t-1] <= replace_na(elec_data_y.loc[c,"Ramp_rate"],10)  *output_data_y.loc[c,'cap_stock'])
                model.p_ramp_cons.add(expr =  model.p[c,t-1] - model.p[c,t] <= replace_na(elec_data_y.loc[c,"Ramp_rate"],10) *output_data_y.loc[c,'cap_stock'])
 
    # constraint on annual generation 
    model.p_potential_g_cons=ConstraintList()
    for c in tech:
        if c not in ['PV','HydroRoR','OnshoreWind', 'OffshoreWind']:
            model.p_potential_g_cons.add(expr = sum(sum(model.p[c,t] for t in Snapshots_h_s[i]) * 365 * size_cluster_ratio_annual[i]/1000 for i in range(No_ed)) <= replace_na(elec_data_y.loc[c,"Potential_annual"],100000))
    
    
    model.line_p_potential_g_cons= ConstraintList()
    model.line_p_potential_g_cons.add(expr = sum(sum(model.line_p['Line',t] for t in Snapshots_h_s[i]) * 365 * size_cluster_ratio_annual[i]/1000 for i in range(No_ed))<=Max_historical_exchange*Domestic_NEP[y-1990])
    
    
    model.line_p_potential_g_cons.add(expr = sum(sum(model.line_p['Line',t] for t in Snapshots_h_s[i]) * 365 * size_cluster_ratio_annual[i]/1000 for i in range(No_ed))>=-Max_historical_exchange*Domestic_NEP[y-1990])

    
    
    ##Optmization
    #Variable costs                      
    var_cost=elec_data_y["Fuel_cost_fuel"]/elec_data_y["Fuel_efficiency"]+elec_data_y["Variable_OM"] #eur2019/MWh or million eu2019/TWh
    annual_var_cost=sum(var_cost.loc[c]*sum(sum(model.p[c,t]/1000 for t in Snapshots_h_s[i]) 
                        *365*size_cluster_ratio_annual[i] for i in range(No_ed)) for c in tech)  #million eur2019/TWh
    
    
    exp_profit=country_data_y["Export_profit"]*sum(sum(model.exp_p['Export', t]/1000 for t in Snapshots_h_s[i])
                                                       *365*size_cluster_ratio_annual[i] for i in range(No_ed))
    
    
    #exp_profit=var_cost.loc['Import']*0.95*sum(sum(model.exp_p['Export', t]/1000 for t in Snapshots_h_s[i])*
    #                                                365*size_cluster_ratio_annual[i] for i in range(No_ed))
    #import costs
    imp_cost=var_cost.loc['Import']*sum(sum(model.imp_p['Import',t]/1000  for t in Snapshots_h_s[i])
                                        *365*size_cluster_ratio_annual[i] for i in range(No_ed))
                                        #million eur2019/TWh 
    #storage costs                                                 
    stor_cost=var_cost.loc['Storage']*sum(sum(model.stor_d['Storage',t]/1000  for t in Snapshots_h_s[i])
                                          *365*size_cluster_ratio_annual[i] for i in range(No_ed))#million eur2019/TWh
   
    #Objective
    model.cost = Objective(expr = annual_var_cost-exp_profit+imp_cost+stor_cost, sense = minimize)
     
    #Optimization
    solver = SolverFactory('gurobi', solver_io="python")
    solver.solve(model, report_timing=True) # solves and updates instance
    #model.pprint()
       
    ##Debug 
    try:
        print(value(model.cost))
         
    except:
        print('***Error, writing .lp file')
        model.write('debug.lp', format="lp", io_options={"symbolic_solver_labels": True})
        
    ##Extraction results
    #Hourly_generation
    possible_tech =['HardCoal','BrownCoal','Gas','Oil','HydroRoR','HydroDam','Nuclear','Geothermal','OnshoreWind','OffshoreWind','PV',
               'Biomass','WasteIncineration','LandfillGas','Biogas','Storage','Import','Export','Storage_charging','Import/Export']
    p_by_tech = pd.DataFrame()
    
    for c in tech :
        for t in Snapshots_h:
            p_by_tech.loc[t,c] = model.p[c,t].value
             
    for t in Snapshots_h:
        for c in possible_tech:
            if c in tech:
                hourly_output_data.loc[(y,c),t]=p_by_tech.loc[t,c]
    
    for t in Snapshots_h:
                                                                                
        p_by_tech.loc[t,'Import'] = model.imp_p['Import',t].value
        p_by_tech.loc[t,'Export'] = -model.exp_p['Export',t].value
        p_by_tech.loc[t,'Storage'] = model.stor_d['Storage',t].value
        p_by_tech.loc[t,'Storage_charging'] = -model.stor_c['Storage_charging',t].value                                                                        
        hourly_output_data.loc[(y,'Import'),t]=p_by_tech.loc[t,'Import']
        hourly_output_data.loc[(y,'Export'),t]=p_by_tech.loc[t,'Export']
        hourly_output_data.loc[(y,'Storage_charging'),t]=p_by_tech.loc[t,'Storage_charging']
        hourly_output_data.loc[(y,'Storage'),t]=p_by_tech.loc[t,'Storage']                                                                       
        
    #Yearly Generation in TWh
    for c in tech:
        output_data.loc[(y,c),'elec_gen']= sum(sum(model.p[c,t].value for t in Snapshots_h_s[i])*365*size_cluster_ratio_annual[i]/1000 for i in range(No_ed))
        
    output_data.loc[(y,'Import'),'elec_gen']=sum(sum(model.imp_p['Import',t].value for t in Snapshots_h_s[i])*365*size_cluster_ratio_annual[i]/1000 for i in range(No_ed))
                                                                                
    export_data.loc[y,'Export']=-1*(sum(sum(model.exp_p['Export',t].value for t in Snapshots_h_s[i])*365*size_cluster_ratio_annual[i]/1000 for i in range(No_ed)))
        
    output_data.loc[(y,'Storage'),'elec_gen']=sum(sum(model.stor_d['Storage',t].value for t in Snapshots_h_s[i])*365*size_cluster_ratio_annual[i]/1000 for i in range(No_ed))
    
    charge_data.loc[y,'Storage_charging']=-1*(sum(sum(model.stor_c['Storage_charging',t].value for t in Snapshots_h_s[i])*365*size_cluster_ratio_annual[i]/1000 for i in range(No_ed)))
    
    global soc_inter
    soc_inter=model.soc_inter.extract_values()
        
