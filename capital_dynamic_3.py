#Code author: Vivien Fisch-Romito, University of Geneva
#Date of modification: 20/02/23
#


import numpy as np
import pandas as pd
#from pyomo.environ import *


# FUNCTIONS USED
def load_seg(Proj_HourlyDem, nb_seg):
    """Splitting projected load curve in different load segments (input for capacity expansion)
    I: Load demand curve projected, number of segments targeted
    O: Load segments"""

    load_segments = np.zeros((nb_seg + 1, 8760))
    load_segments[0, :] = min(Proj_HourlyDem)

    cap_seg = min(Proj_HourlyDem) + np.array(range(1, nb_seg + 1)) / nb_seg * (
        max(Proj_HourlyDem) - min(Proj_HourlyDem)
    )
    print("min proj dem", min(Proj_HourlyDem))
    cap_seg = np.insert(cap_seg, 0, min(Proj_HourlyDem))
    print("cap_seg", cap_seg)

    for i in range(1, nb_seg + 1):
        load_segments[i, :] = (
            np.where(Proj_HourlyDem > cap_seg[i], cap_seg[i], Proj_HourlyDem)
            - cap_seg[i - 1]
        )
    load_segments = np.where(load_segments < 0, 0, load_segments)

    return load_segments


def discount_fact(discount, lspan):
    """Creating an array of discount factors over the lifetime of an asset
    I: discount factor,lifespan
    O: discount factors"""

    df = np.geomspace((1 + discount) ** -1, (1 + discount) ** -lspan, num=int(lspan))
    return df


def tau_growth(actual_value, old_value, time):
    """Calculating the average growth rate
    I: discount factor,lifespan
    O: discount factors"""
    if actual_value == 0 or old_value == 0:
        tau = 0
    else:
        tau = (actual_value / old_value) ** (1 / time) - 1
    return tau


def ret_vint(df, year):
    """Modifying the dataframe with installed capacity and commitment year when capacity is retired
    I: dataframe, actual year minus lifetime of technology
    O: capacity committed on the year considered"""
    try:
        cap = df.loc[year].copy()
        df.loc[year] = 0
    except:
        cap = 0
    return cap


def ret_vint_exp(df, year):
    """Checking the dataframe with installed capacity and commitment year to evaluate projected future retired capacity
    I: dataframe, actual year minus lifetime of technology
    O: capacity committed on the year considered"""
    try:
        cap = df.loc[year]
    except:
        cap = 0
    return cap


def early_ret_vint(early_ret, vint_cap, new_inst_cap):
    """Modification of vintage dataframe in case of earlier retirement of capacity
    I: dataframe, actual year minus lifetime of technology
    O: capacity committed on the year considered"""
    c = new_inst_cap.copy()

    while early_ret > 0:
        
        vint_cap.loc[vint_cap<1e-08]=0
        c.loc[c<1e-08]=0
        print('test c',c.sum())
        if vint_cap.values.sum() == 0 and c.values.sum()==0:
            print('no cap remaining')
            early_ret=0
            continue
        
        print('early ret function', early_ret)
        
        if vint_cap.sum() == 0:
            if (early_ret - c.loc[(c.loc[c > 0]).index[0]]) > 0:
                early_ret -= c.loc[(c.loc[c > 0]).index[0]]
                c.loc[(c.loc[c > 0]).index[0]] = 0

            else:
                c.loc[(c.loc[c > 0]).index[0]] -= early_ret
                early_ret = 0

        else:
            if (early_ret - vint_cap.loc[(vint_cap.loc[vint_cap > 0]).index[0]]) > 0:
                early_ret -= vint_cap.loc[(vint_cap.loc[vint_cap > 0]).index[0]]
                vint_cap.loc[(vint_cap.loc[vint_cap > 0]).index[0]] = 0

            else:
                vint_cap.loc[(vint_cap.loc[vint_cap > 0]).index[0]] -= early_ret
                early_ret = 0
                
        

    return vint_cap.values, early_ret, c.values


def ext_cap(vint_cap, y, sh_ext):
    """Modifying the dataframe tracking capacity vintage in case of lifetime extension
    I: vintage dataframe, commitment year, share of installed capacity extended
    O: closed capital"""

    try:

        close_c = (1 - sh_ext) * vint_cap.loc[y]
        print("close_c", close_c)
        vint_cap.loc[y] = vint_cap.loc[y] - close_c
        print("Vintage stock found ")
        print(vint_cap.loc[y], "GW extended")
    except:
        print("No vintage stock")
        close_c = 0
    return close_c


def share_inv(tech, lbda, price_tech, drop_tech):
    """Multinomial logit equation for investment
    I: vintage dataframe, commitment year, share of installed capacity extended
    O: closed capital"""
    nom=price_tech.loc[tech]**(-lbda)
    denom = sum(np.power(price_tech.drop(labels=drop_tech, axis=0).values, -lbda))
    ms = nom / denom
    return ms
    
    
def alcoe(i, f, v, fuel_eff, fprice, discount, lspan, lf, df):
    """Annualized levelized cost
    I: investment cost, fixed cost,O&M cost, fuel efficiency, fuel cost, discount rate,
    life span, technology load factor, duration factor (annual electricity generation ration for the investment segment)
    O: levelized cost"""
    if df == 0:
        df = 0.001
    inv_cost = (
        i * (discount / (1 - (1 + discount) ** -lspan)) / (24 * 365 * lf * df)
    )
    fixed_cost = f / (24 * 365 * lf * df)
    var_cost = v/1000
    fuel_cost = 1 / fuel_eff * fprice/1000
    cost = inv_cost + fixed_cost + var_cost + fuel_cost
    return cost  #eur2019/kWh


def lcoe(i, f, v, fuel_eff, fprice, discount, lspan, lf, df):
    if df == 0:
        df = 0.001
    inv_cost = (i) / sum(24 * 365 * lf * df * discount_fact(discount, lspan))  #EUR2019/kWh   
    fixed_cost = sum(f * discount_fact(discount, lspan)) / sum(
        24 * 365 * lf * df * discount_fact(discount, lspan)  #EUR2019/kWh   
    )
    var_cost = sum(v/1000 * 24 * 365 * lf * df * discount_fact(discount, lspan)) / sum(
        24 * 365 * lf * df * discount_fact(discount, lspan)  #EUR2019/kWh   
    )
    fuel_cost = sum(
        1 / fuel_eff * fprice/1000 * 24 * 365 * lf * df * discount_fact(discount, lspan)
    ) / sum(24 * 365 * lf * df * discount_fact(discount, lspan)) #EUR2019/kWh   

    cost = inv_cost + fixed_cost + var_cost + fuel_cost
    return cost


def replace_na(a, b):
    if pd.isna(a):
        r = b
    else:
        r = a
    return r


def main(
    y,
    country,
    country_data,
    elec_data,
    output_data,
    proj_rep_days,
    study_frame,
    hcoal_init,
    gas_init,
    nuke_init,
    bcoal_init,
    soc_nuc,
    soc_wind,
    soc_coal,
    soc_gas,
    soc_pv,
    oecd_gov,
    wacc
):

    # A. RUN FRAMEWORK

    # Extract scenario alternatives from studyframe file
    ind_lock = study_frame.loc["ind_lock"]
    ind_trans = study_frame.loc["ind_trans"]
    ind_bound = study_frame.loc["ind_bound"]
    ind_finance = study_frame.loc["ind_finance"]
    ind_social = study_frame.loc["ind_social"]
    ind_gov = study_frame.loc["ind_gov"]
    
    elec_data_run=elec_data.copy()

    # technology names
    tech = elec_data_run.index.get_level_values(1).unique()
    # renewables tech names
    if "OffshoreWind" in tech:
        renew_tech = ["PV", "HydroRoR", "OnshoreWind", "OffshoreWind"]
    else:
        renew_tech = ["PV", "HydroRoR", "OnshoreWind"]

    # Parameters definition, sometimes based on societal factors integration
    
    # Number of load demand curve segments to allocate investments (peak, intermediate, baseload...)
    nb_seg = 4
    
    # Cost sensitivity for multinomial logit investment equation
    if ind_bound == 1:
        lbda = 5
    else:
        lbda = 30
    
    # Choice of discount rate for levelized costs calculation
    discount = pd.Series(data=0.035, index=tech)
    if ind_finance == 1 and y >= 2009:
        wacc=wacc.loc[country]
        for c in discount.index:
            if c in wacc.columns:
                discount[c] = wacc.loc[y,c]
            else:
                discount[c] = np.mean(wacc.loc[y,:].values)
    print("DISCOUNT", discount)

    # Upper limit of installed capacity for each technology (either directly or combining the maximum annual generation and minimum load factor)
    max_cap = np.minimum(
        (
            elec_data_run.loc[y + 1, "Potential_annual"]
            * 1000
            / (8760 * elec_data_run.loc[y + 1, "LF_min"])
        ).fillna(10**6),
        elec_data_run.loc[y + 1, "Potential_installed"].fillna(10**6),
    )
    
    #lifetime extension of installed power capacity when lock-in
    lifetime_ext = 10
     
    # Index with technologies not considered because maximum capacity already reached
    index_max = []
    for c in tech:
        if output_data.loc[(y, c), "cap_stock"] >= max_cap.loc[c]:
            index_max.append(c)

    # Parameter defining the perception of investment costs (i.e adoption costs) for wind because of public acceptance
    soc_wind_push = 1
    soc_pv_push = 1
    gov_wind_push = 1
    gov_pv_push = 1
    loc_yes_wind_ratio = 0.65
    soc_no_threshold = 60
    soc_push_threshold = 80

    # Effet of public acceptance on technologies considered and adoption costs
    if ind_social == 1:
        if country in soc_nuc.index.get_level_values(0).unique().values:
            if pd.isna(soc_nuc.loc[country, y]):
                no_nuc = soc_nuc.loc[country, soc_nuc.loc[country].first_valid_index()]
            else:
                no_nuc = soc_nuc.loc[country, y]
            if no_nuc >= soc_no_threshold:
                index_max.append("Nuclear")

        if (country in soc_wind.index.get_level_values(0).unique().values) and (
            y > 1999
        ):
            if pd.isna(soc_wind.loc[country, y]):
                yes_wind = soc_wind.loc[soc_wind.first_valid_index()]
            else:
                yes_wind = soc_wind.loc[country, y]

            if yes_wind * loc_yes_wind_ratio < (100 - soc_no_threshold):
                index_max.append("OnshoreWind")
                if "OffshoreWind" in tech:
                    index_max.append("OffshoreWind")
            elif yes_wind >= soc_push_threshold:
                soc_wind_push = 0.8
            else:
                soc_wind_push = 1

        if (country in soc_coal.index.get_level_values(0).unique().values) and (
            y > 1999
        ):
            if pd.isna(soc_coal.loc[country, y]):
                no_coal = soc_coal.loc[
                    country, soc_coal.loc[country].first_valid_index()
                ]
            else:
                no_coal = soc_coal.loc[country, y]
            if no_coal >= soc_no_threshold:
                index_max.append("HardCoal")
                if "BrownCoal" in tech:
                    index_max.append("BrownCoal")

        if (country in soc_gas.index.get_level_values(0).unique().values) and (
            y > 1999
        ):
            if pd.isna(soc_gas.loc[country, y]):
                no_gas = soc_gas.loc[country, soc_gas.loc[country].first_valid_index()]
            else:
                no_gas = soc_gas.loc[country, y]
            if no_gas >= soc_no_threshold:
                index_max.append("Gas")

        if (country in soc_pv.index.get_level_values(0).unique().values) and (y > 1999):
            if pd.isna(soc_pv.loc[country, y]):
                yes_pv = soc_pv.loc[soc_pv.first_valid_index()]
            else:
                yes_pv = soc_pv.loc[country, y]

            if yes_pv < (100 - soc_no_threshold):
                index_max.append("PV")
            elif yes_pv >= soc_push_threshold:
                soc_pv_push = 0.8
            else:
                soc_pv_push = 1
    
    #Effect of governance on adoption costs of renewable technologies
    if ind_gov == 1:
        if country in oecd_gov.index.get_level_values(0).unique().values:
            if (oecd_gov.loc[(country, min(y,2013)), "elec_entry"] <= 2) and (
                oecd_gov.loc[(country, min(y,2013)), "elec_pub_own"] >= 4
            ):
                gov_pv_push = 0.8
                gov_wind_push = 0.8
            elif (oecd_gov.loc[(country, min(y,2013)), "elec_entry"] >= 4) and (
                oecd_gov.loc[(country, min(y,2013)), "elec_pub_own"] <= 2
            ):
                gov_pv_push = 1.2
                gov_wind_push = 1.2
            else:
                pass
        else:
            pass
            
    elec_data_run.loc[(y, "OnshoreWind"), "Inv"] = (
        elec_data_run.loc[(y, "OnshoreWind"), "Inv"] * soc_wind_push * gov_wind_push
    )
    if "OffshoreWind" in tech:
        elec_data_run.loc[(y, "OffshoreWind"), "Inv"] = (
            elec_data_run.loc[(y, "OffshoreWind"), "Inv"] * soc_wind_push * gov_wind_push
        )
    elec_data_run.loc[(y, "PV"), "Inv"] = (
        elec_data_run.loc[(y, "PV"), "Inv"] * soc_pv_push * gov_pv_push
    )

    print("Technologies where the maxim cap is reached", index_max)

    print("actual stock_hcoal", output_data.loc[(y, "HardCoal"), "cap_stock"])
    print(
        "vintage stock_hcoal",
        hcoal_init.values.sum()
        + output_data.loc[pd.IndexSlice[:y, "HardCoal"], "cap_new"].values.sum(),
    )
    print("actual stock_gas", output_data.loc[(y, "Gas"), "cap_stock"])
    print(
        "vintage stock_gas",
        gas_init.values.sum()
        + output_data.loc[pd.IndexSlice[:y, "Gas"], "cap_new"].values.sum(),
    )
    
    if 'BrownCoal' in tech:
        print("actual stock_bcoal", output_data.loc[(y, "BrownCoal"), "cap_stock"])
        print(
            "vintage stock_bcoal",
            bcoal_init.values.sum()
            + output_data.loc[pd.IndexSlice[:y, "BrownCoal"], "cap_new"].values.sum(),
        )
    
    # B.EARLIER RETIREMENT OF CAPACITY
    if ind_trans == 1:
        #Comparing variable cost of incumbent technology with lcoe of alternative technologies
        df_elec = output_data.loc[(y,), "elec_gen"] / (
            output_data.loc[(y,), "cap_stock"]
            * elec_data_run.loc[(y,), "LF_max"]
            * 8760
            / 1000
        )
        df_elec[df_elec == 0] = 0.0001
        print("df_elec_updated", df_elec)
        
        op_cost = (
            elec_data_run.loc[y, "Fuel_cost_fuel"] / elec_data_run.loc[y, "Fuel_efficiency"]
            + elec_data_run.loc[y, "Variable_OM"]
        )/1000 + elec_data_run.loc[y, "Fixed_OM_annual"]/(24 * 365 * elec_data_run.loc[y, "LF_max"]*df_elec)

        for c in ["Nuclear", "BrownCoal", "Gas", "HardCoal"]:
            if c in tech:
                corresp_lcoe = pd.Series(
                    data=alcoe(
                        elec_data_run.loc[y, "Inv"].values,
                        elec_data_run.loc[y, "Fixed_OM_annual"].values,
                        elec_data_run.loc[y, "Variable_OM"].values,
                        elec_data_run.loc[y, "Fuel_efficiency"].values,
                        elec_data_run.loc[y, "Fuel_cost_fuel"].values,
                        discount,
                        elec_data_run.loc[y, "Lifetime"].values,
                        elec_data_run.loc[y, "LF_max"].values,
                        df_elec.loc[c],
                    ),
                    index=tech,
                )
                corresp_lcoe.loc[c] = op_cost.loc[c]
                corresp_lcoe[corresp_lcoe < 0] = 0
                print(c,' corresp_lcoe ', corresp_lcoe)

                index_max_temp = np.unique(index_max).tolist()
                if c in index_max_temp:
                    index_max_temp.remove(c)

                early_ret = (
                    1
                    - share_inv(
                        c,
                        30,
                        corresp_lcoe,
                        np.concatenate((["Import", "Storage"], index_max_temp)),
                    )
                ) * output_data.loc[(y, c), "cap_stock"]
                early_ret = np.nan_to_num(early_ret)
                
                print("earlyret", c, early_ret)

                # if early_ret<min_cap.loc[c,1]:
                #    early_ret=0
                # output_data.loc[(y+6,c),'cap_closed']+=early_ret

                # Limitation on the amount of capacity that can be retired every year (2% of total capacity for each tech)
                early_ret = min(early_ret, 0.02 * output_data.loc[y, "cap_stock"].sum())

                output_data.loc[(y + 1, c), "cap_closed"] += early_ret
                print(c, " ", early_ret, "GW will be early retired")

                if c == "HardCoal":
                    print("hcoal_init before retirement", hcoal_init)
                    (
                        hcoal_init[:],
                        early_ret,
                        output_data.loc[pd.IndexSlice[:, "HardCoal"], "cap_new"]
                    ) = early_ret_vint(
                        early_ret,
                        hcoal_init,
                        output_data.loc[pd.IndexSlice[:, "HardCoal"], "cap_new"]
                    )
                    print("hcoal_init after retirement", hcoal_init)

                elif c == "BrownCoal":
                    (
                        bcoal_init[:],
                        early_ret,
                        output_data.loc[pd.IndexSlice[:, "BrownCoal"], "cap_new"],
                    ) = early_ret_vint(
                        early_ret,
                        bcoal_init,
                        output_data.loc[pd.IndexSlice[:, "BrownCoal"], "cap_new"]
                    )

                elif c == "Gas":
                    (
                        gas_init[:],
                        early_ret,
                        output_data.loc[pd.IndexSlice[:, "Gas"], "cap_new"],
                    ) = early_ret_vint(
                        early_ret,
                        gas_init,
                        output_data.loc[pd.IndexSlice[:, "Gas"], "cap_new"]
                    )
                else:
                    (
                        nuke_init[:],
                        early_ret,
                        output_data.loc[pd.IndexSlice[:, "Nuclear"], "cap_new"]
                    ) = early_ret_vint(
                        early_ret,
                        nuke_init,
                        output_data.loc[pd.IndexSlice[:, "Nuclear"], "cap_new"]
                    )

    # C.CAPACITY NEEDED TO BALANCE DEMAND IN 5 YEARS
    # Evaluation of expecting capacity remaining in 5 years
    exp_cap = output_data.loc[y, "cap_stock"].copy()

    for i in range(y + 1, y + 6):
        # creating index of year-lifetime for retirement of new capacity built on the 1990-2020 period
        lifetime_index = (
            (i - (elec_data_run.loc[y, "Lifetime"]))
            .astype(int)
            .reset_index()
            .reindex(columns=["Lifetime", "Techno"])
        )
        lifetime_index["Lifetime"] = np.where(
            (lifetime_index.Lifetime < 1990), 1990, lifetime_index.Lifetime
        )
        lifetime_index = lifetime_index.values.tolist()

        # For all capacities, retired capacities is initially defined indexing on 1990 capacity
        if i < 2020:
            closed_cap = (
                elec_data_run.loc[i, "Initial_retired_capacity"].copy()
                + output_data.loc[i, "cap_closed"].copy()
            )
        else:
            closed_cap = pd.Series(0, index=tech)

        # For Nuclear, Gas and Coal, endogeneous retirement approach based on year of commitment
        if i == 1991:
            print("closed_cap_expected", closed_cap)
            closed_cap.loc["Gas"] = gas_init.loc[
                : i - elec_data_run.loc[(y, "Gas"), "Lifetime"]
            ].values.sum()
            closed_cap.loc["HardCoal"] = hcoal_init.loc[
                : i - elec_data_run.loc[(y, "HardCoal"), "Lifetime"]
            ].values.sum()
            closed_cap.loc["Nuclear"] = nuke_init.loc[
                : i - elec_data_run.loc[(y, "Nuclear"), "Lifetime"]
            ].values.sum()

            if "BrownCoal" in tech:
                closed_cap.loc["BrownCoal"] = bcoal_init.loc[
                    : i - elec_data_run.loc[(y, "BrownCoal"), "Lifetime"]
                ].values.sum()

        else:
            closed_cap.loc["Gas"] = ret_vint_exp(
                gas_init, int(i - elec_data_run.loc[(y, "Gas"), "Lifetime"])
            ) + ret_vint_exp(
                gas_init, int(y + 1 - elec_data_run.loc[(y, "Gas"), "Lifetime"]) - lifetime_ext
            )

            closed_cap.loc["HardCoal"] = ret_vint_exp(
                hcoal_init, int(i - elec_data_run.loc[(y, "HardCoal"), "Lifetime"])
            ) + ret_vint_exp(
                hcoal_init, int(i - elec_data_run.loc[(y, "HardCoal"), "Lifetime"]) - lifetime_ext
            )

            closed_cap.loc["Nuclear"] = ret_vint_exp(
                nuke_init, int(i - elec_data_run.loc[(y, "Nuclear"), "Lifetime"])
            ) + ret_vint_exp(
                nuke_init, int(i - elec_data_run.loc[(y, "Nuclear"), "Lifetime"]) - lifetime_ext
            )

            if "BrownCoal" in tech:
                closed_cap.loc["BrownCoal"] = ret_vint_exp(
                    bcoal_init, int(i - elec_data_run.loc[(y, "BrownCoal"), "Lifetime"])
                ) + ret_vint_exp(
                    bcoal_init, int(i - elec_data_run.loc[(y, "BrownCoal"), "Lifetime"]) - lifetime_ext
                )

        for c in ["Gas", "HardCoal", "Nuclear", "BrownCoal"]:
            if c in tech:
                closed_cap.loc[c] += output_data.loc[(i, c), "cap_closed"]

        # Adding the capacitiy built during the modelling period and retired after lifetime
        closed_cap.update(
            closed_cap + output_data.loc[lifetime_index, "cap_new"].values
        )
        new_cap = output_data.loc[i, "cap_new"].copy()
        print("closed_cap", closed_cap)
        exp_cap = exp_cap - closed_cap + new_cap
    print("exp_cap", exp_cap)
    print("expected capital in 5 years", sum(exp_cap))
    
    

    # Expected required capacity in 5 years
    # Get the projected peak of demand
    Proj_HourlyDem = proj_rep_days.loc["HourlyDem"]
    print('peak in capital dynamic file', max(Proj_HourlyDem))
    
    # peak expected with margin factor integrated
    peak_exp = max(Proj_HourlyDem) * (1 + country_data.loc[y,'Capacity_margin'])
  

    # peak equation to get the expected required capital
    elec_data_run.loc[(y, 'OnshoreWind'), "Peak_contr"]=0.15
    if 'OffshoreWind' in tech:
        elec_data_run.loc[(y, 'OffshoreWind'), "Peak_contr"]=0.15
    
    exp_req_cap = (
        sum(exp_cap)
        * peak_exp
        / sum(
            exp_cap.loc[c]
            * min(elec_data_run.loc[(y, c), "Peak_contr"], elec_data_run.loc[(y, c), "LF_max"])
            * (1 - elec_data_run.loc[(y, c), "Own_use"])
            for c in tech
        )
    )
    print("expected required capital in 5 years", exp_req_cap)

    # Total new capital to order
    tot_order_cap = max(exp_req_cap - sum(exp_cap), 0)
    print("total capital to order", tot_order_cap)

    # Constraint added to avoid a too high dependence on import capacity (case of Estonia)
    if exp_cap.loc["Import"] > 0.5 * sum(exp_cap):
        print("too strong dependence on imports")
        tot_order_cap = max(exp_cap.loc["Import"], tot_order_cap)
        
        
        
    
    # D.SPLITTING TOTAL REQUIRED CAPACITY TO ORDER

    #D.1 Investments in non dispatchable renewables technologies
    
    # Net load demand curve : substracting contribution of expected renewables capacity in 5 years
    res_gen = 0
    for c in tech:
        if c in renew_tech:
            res_gen += proj_rep_days.loc[c].values * exp_cap.loc[c]
            

    print("Proj_Hourly_dem", Proj_HourlyDem)
    Proj_HourlyDem = Proj_HourlyDem - res_gen
    print("Proj_Hourly_dem after contribution of remaining RES", Proj_HourlyDem)
    Proj_HourlyDem[Proj_HourlyDem < 0] = 0

    proj_rep_days.loc["HourlyDem"] = Proj_HourlyDem
    proj_rep_days = proj_rep_days.sort_values(by="HourlyDem", axis=1, ascending=False)
    Proj_HourlyDem = proj_rep_days.loc["HourlyDem"]
    
    
    # Making the model stop investing in base load segment if sum tech times min LF>min load
    if sum(exp_cap.loc[c] * elec_data_run.loc[(y, c), "LF_min"] for c in tech) > min(
        Proj_HourlyDem
    ):
        no_base = 1
        off_base = 0

    else:
        no_base = 0
        off_base = 1
    
    
    # Splitting projected net load demand curve in load segments with different duration factor (df segment=1 for baseload)
    load_segments = load_seg(Proj_HourlyDem, nb_seg)
    print("load_segment to allocate res", load_segments)

    # Get the duration factor of each segment to calculate specific lcoe
    global df_segments
    df_segments = np.sum(load_segments, axis=1) / (8760 * load_segments[:, 0])
    df_segments = np.nan_to_num(df_segments)
    print("duration factor", df_segments)

    # LCOE calculation for each segment
    lcoe_segments = pd.DataFrame(data=np.zeros((len(tech), nb_seg + 1)), index=tech)

    # Calculating the different components of technology costs per segments and technology
    for i in range(0, nb_seg + 1):
        lcoe_segments.iloc[:, i] = alcoe(
            elec_data_run.loc[y, "Inv"].values,
            elec_data_run.loc[y, "Fixed_OM_annual"].values,
            elec_data_run.loc[y, "Variable_OM"].values,
            elec_data_run.loc[y, "Fuel_efficiency"].values,
            elec_data_run.loc[y, "Fuel_cost_fuel"].values,
            discount,
            elec_data_run.loc[y, "Lifetime"].values,
            elec_data_run.loc[y, "LF_max"].values,
            df_segments[i],
        )

    # modelling patch for negative lcoe (because of negative variable costs from data such as waste incineration in GBR
    lcoe_segments[lcoe_segments < 0] = 0.0000001
    
    #Modelling patch in case of unused load segments because of too low demand
    size_seg = load_segments[:, 0]
    if size_seg[0] == 0:
        lcoe_segments.iloc[:, 0] = 1.0

    print("lcoe_segments", lcoe_segments)




    # Projected required capital in each segment based on lcoe
    order_cap = 0 * lcoe_segments.copy()
    # Splitting required capacity per load band and technology. The total needed capacity is splitted based on the relative size of each load band size_seg[i]/sum(size_seg)    

    # fictitious competition between all technologies to allocate renewables to one load band
    print("size_seg", size_seg)
    for i in range(0, nb_seg + 1):
        for c in renew_tech:
            if i > 1:
                order_cap.loc[c, i] = (
                    tot_order_cap
                    * size_seg[i]
                    / (sum(size_seg) - no_base * (size_seg[0] + size_seg[1]))
                    * share_inv(
                        c,
                        lbda,
                        lcoe_segments.loc[:, i],
                        np.concatenate((["Nuclear", "Storage", "Import"], index_max)),
                    )
                )

            else:
                order_cap.loc[c, i] = (
                    off_base
                    * tot_order_cap
                    * size_seg[i]
                    / (sum(size_seg))
                    * share_inv(
                        c,
                        lbda,
                        lcoe_segments.loc[:, i],
                        np.concatenate((["Import", "Storage"], index_max)),
                    )
                )

            order_cap.loc[c, order_cap.columns != np.argmax(order_cap.loc[c])] = 0

    for c in renew_tech:
        if c in index_max:
            print("MAX RENEW CAP REACHED")
            order_cap.loc[c, :] = 0

    renew_order_cap = order_cap.copy()
    print("renew_order_cap", renew_order_cap)


    # D.2 Investments in dispatchable technologies
    
    # Residual load demand curve by substracting future non dispatchable renewable capacity generation 
    Residual_HourlyDem = Proj_HourlyDem.copy()
    print("Load_demand_curve", Residual_HourlyDem)
    for c in renew_tech:
        Residual_HourlyDem = (
            Residual_HourlyDem - sum(order_cap.loc[c, :]) * proj_rep_days.loc[c].values
        )
    Residual_HourlyDem[Residual_HourlyDem < 0] = 0
    print("Residual Load demand curve", Residual_HourlyDem)

    Residual_HourlyDem = Residual_HourlyDem.sort_values(ascending=False)

    # New segments division
    rldc_load_segments = load_seg(Residual_HourlyDem, nb_seg)

    # Get the duration factor of each segment to calculate specific lcoe
    global rldc_df_segments
    rldc_df_segments = np.sum(rldc_load_segments, axis=1) / (
        8760 * rldc_load_segments[:, 0]
    )
    rldc_df_segments = np.nan_to_num(rldc_df_segments)
    print("rldc_df_segments", rldc_df_segments)

    # LCOE calculation for each segment
    rldc_lcoe_segments = pd.DataFrame(
        data=np.zeros((len(tech), nb_seg + 1)), index=tech
    )

    for i in range(0, nb_seg + 1):
        rldc_lcoe_segments.iloc[:, i] = alcoe(
            elec_data_run.loc[y, "Inv"].values,
            elec_data_run.loc[y, "Fixed_OM_annual"].values,
            elec_data_run.loc[y, "Variable_OM"].values,
            elec_data_run.loc[y, "Fuel_efficiency"].values,
            elec_data_run.loc[y, "Fuel_cost_fuel"].values,
            discount,
            elec_data_run.loc[y, "Lifetime"].values,
            elec_data_run.loc[y, "LF_max"].values,
            rldc_df_segments[i],
        )

    # modelling patch for negative lcoe (waste incineration in GBR)
    rldc_lcoe_segments[rldc_lcoe_segments < 0] = 0.0000001

    print("rldc_lcoe_segments", rldc_lcoe_segments)

    #Modelling patch in case of unused load segments because of too low demand
    rldc_size_seg = rldc_load_segments[:, 0]
    print("rldc_size_seg", rldc_size_seg)
    if rldc_size_seg[0] == 0:
        rldc_lcoe_segments.iloc[:, 0] = 1.0


    order_cap = 0 * rldc_lcoe_segments.copy()


    # Remain capacity to order after non dispatchable renewable investments
    remain_order_cap = tot_order_cap - sum(
        sum(renew_order_cap.loc[c]) for c in renew_tech
    )
    print("remain order cap", remain_order_cap)
    
    # Transmission capacity added only to satisfy the peak (investment for last load segment only)
    order_cap.loc["Import", 4] = (
        remain_order_cap
        * rldc_size_seg[4]
        / (sum(rldc_size_seg) - no_base * (rldc_size_seg[0] + rldc_size_seg[1]))
        * share_inv(
            "Import",
            lbda,
            rldc_lcoe_segments.loc[:, 4],
            np.concatenate((["Nuclear", "Storage"], renew_tech, index_max)),
        )
    )

    #Constraint on installed capacity of transmission capacity
    if (exp_cap.loc["Import"] + order_cap.loc["Import", 4]) > max_cap.loc["Import"]:
        order_cap.loc["Import", 4] = max(
            max_cap.loc["Import"] - exp_cap.loc["Import"], 0
        )


    #Splitting remaining capacity needed, excluding variable renewables capacity, storage and transmission capacity
    for i in range(0, nb_seg + 1):
        for c in tech.drop(
            labels=np.concatenate((renew_tech, ["Import", "Storage"], index_max))
        ):

            if i > 1:
                order_cap.loc[c, i] = (
                    remain_order_cap
                    * rldc_size_seg[i]
                    / (
                        sum(rldc_size_seg)
                        - no_base * (rldc_size_seg[0] + rldc_size_seg[1])
                    )
                    - order_cap.loc["Import", i]
                ) * share_inv(
                    c,
                    lbda,
                    rldc_lcoe_segments.loc[:, i],
                    np.concatenate(
                        (["Nuclear", "Storage", "Import"], renew_tech, index_max)
                    ),
                )

            else:
                #Nuclear only for load segments with high duration factor (base load like)
                order_cap.loc[c, i] = (
                    off_base
                    * remain_order_cap
                    * rldc_size_seg[i]
                    / sum(rldc_size_seg)
                    * share_inv(
                        c,
                        lbda,
                        rldc_lcoe_segments.loc[:, i],
                        np.concatenate((["Import", "Storage"], renew_tech, index_max)),
                    )
                )

    order_cap.loc["Nuclear", 2 : nb_seg + 1] = 0
    order_cap.loc["Import", 0] = 0
    order_cap.loc["Import", 1] = 0

    print("order_cap_final", sum(order_cap.values))
    print("order_cap", order_cap)

   
   
   
    order_cap = order_cap.sum(axis=1).sort_index()
    renew_order_cap = renew_order_cap.sum(axis=1).sort_index()
    for c in renew_tech:
        order_cap.loc[c] = renew_order_cap.loc[c]

    print("order cap per tech before max cap", order_cap)
    print("sum order cap per tech before max cap", sum(order_cap))

    # Constraint on maximum potential installed 
    excess_cap = 0
    for c in tech:
        if (exp_cap.loc[c] + order_cap.loc[c]) > max_cap.loc[c]:
            excess_cap += max(exp_cap.loc[c] + order_cap.loc[c] - max_cap.loc[c], 0)
            index_max.append(c)
            order_cap.loc[c] = max(max_cap.loc[c] - exp_cap.loc[c], 0)

    # Redistribution of excess capita (above maximum) between other technologies
    excess_order_cap = 0 * rldc_lcoe_segments.copy()
    for i in range(0, nb_seg + 1):
        for c in tech.drop(
            labels=np.concatenate((renew_tech, ["Import", "Storage"], index_max))
        ):

            if i > 1:
                excess_order_cap.loc[c, i] = (
                    excess_cap
                    * rldc_size_seg[i]
                    / (
                        sum(rldc_size_seg)
                        - no_base * (rldc_size_seg[0] + rldc_size_seg[1])
                    )
                ) * share_inv(
                    c,
                    lbda,
                    rldc_lcoe_segments.loc[:, i],
                    np.concatenate(
                        (["Nuclear", "Storage", "Import"], renew_tech, index_max)
                    ),
                )

            else:
                excess_order_cap.loc[c, i] = (
                    off_base
                    * excess_cap
                    * rldc_size_seg[i]
                    / sum(rldc_size_seg)
                    * share_inv(
                        c,
                        lbda,
                        rldc_lcoe_segments.loc[:, i],
                        np.concatenate((["Import", "Storage"], renew_tech, index_max)),
                    )
                )

    excess_order_cap.loc["Nuclear", 2 : nb_seg + 1] = 0
    excess_order_cap.loc["Import", 0] = 0
    excess_order_cap.loc["Import", 1] = 0
    print("excess_order_cap", excess_order_cap)
    excess_order_cap = excess_order_cap.sum(axis=1).sort_index()
    order_cap = order_cap + excess_order_cap

    print("order cap per tech after max cap", order_cap)
    print("sum order cap per tech after max cap", sum(order_cap))




    # Minimum new built capacity  (in order to avoid nuclear reactor of 1kw for instance)
    min_cap = pd.read_csv(
        "./inputs/_common/Datasets/min_capacity.csv", index_col=0, header=None
    )
    for c in tech:
        if order_cap.loc[c] < min_cap.loc[c, 1]:
            order_cap.loc[c] = 0

    # if ind_lock==1:
    #    output_data.loc[y+5,'cap_new']=order_cap.values else

    output_data.loc[y + 1, "cap_new"] = (
        output_data.loc[y + 1, "cap_new"].values + order_cap.values / 5
    )
    output_data.loc[(y + 1, "Nuclear"), "cap_new"] -= order_cap.loc["Nuclear"] / 5
    new_nuke = order_cap.loc["Nuclear"] / 5
    t = 0
    while (new_nuke < min_cap.loc["Nuclear", 1]) and (new_nuke > 0):
        t += 1
        print("t", t)
        new_nuke = order_cap.loc["Nuclear"] / (5 - t)
        print("new_nuke", new_nuke)

    output_data.loc[(y + 1 + t, "Nuclear"), "cap_new"] += new_nuke
    print("test nuke", output_data.loc[(y + 1 + t, "Nuclear"), "cap_new"])


    #D.3 Retirement of the installed capacity
    
    # Capacity retirement due to lifetime
    lifetime_index = (
        (y + 1 - (elec_data_run.loc[y, "Lifetime"]))
        .astype(int)
        .reset_index()
        .reindex(columns=["Lifetime", "Techno"])
    )
    lifetime_index["Lifetime"] = np.where(
        (lifetime_index.Lifetime < 1990), 1990, lifetime_index.Lifetime
    )
    lifetime_index = lifetime_index.values.tolist()

    # Endogeneous retirement for Gas, Coal and Nuclear because of lifetime
    if y == 1990:
        output_data.loc[(y + 1, "Gas"), "cap_closed"] += gas_init.loc[
            : y + 1 - elec_data_run.loc[(y, "Gas"), "Lifetime"]
        ].values.sum()
        gas_init.loc[: y + 1 - elec_data_run.loc[(y, "Gas"), "Lifetime"]] = 0

        output_data.loc[(y + 1, "HardCoal"), "cap_closed"] += hcoal_init.loc[
            : y + 1 - elec_data_run.loc[(y, "HardCoal"), "Lifetime"]
        ].values.sum()
        hcoal_init.loc[: y + 1 - elec_data_run.loc[(y, "HardCoal"), "Lifetime"]] = 0

        output_data.loc[(y + 1, "Nuclear"), "cap_closed"] += nuke_init.loc[
            : y + 1 - elec_data_run.loc[(y, "Nuclear"), "Lifetime"]
        ].values.sum()
        nuke_init.loc[: y + 1 - elec_data_run.loc[(y, "Nuclear"), "Lifetime"]] = 0

        if "BrownCoal" in tech:
            output_data.loc[(y + 1, "BrownCoal"), "cap_closed"] += bcoal_init.loc[
                : y + 1 - elec_data_run.loc[(y, "BrownCoal"), "Lifetime"]
            ].values.sum()
            bcoal_init.loc[: y + 1 - elec_data_run.loc[(y, "BrownCoal"), "Lifetime"]] = 0

    else:
        #no lock-in
        if ind_lock == 0:
            output_data.loc[(y + 1, "Gas"), "cap_closed"] += ret_vint(
                gas_init, int(y + 1 - elec_data_run.loc[(y, "Gas"), "Lifetime"])
            )
            output_data.loc[(y + 1, "HardCoal"), "cap_closed"] += ret_vint(
                hcoal_init, int(y + 1 - elec_data_run.loc[(y, "HardCoal"), "Lifetime"])
            )
            output_data.loc[(y + 1, "Nuclear"), "cap_closed"] += ret_vint(
                nuke_init, int(y + 1 - elec_data_run.loc[(y, "Nuclear"), "Lifetime"])
            )
            if "BrownCoal" in tech:
                output_data.loc[(y + 1, "BrownCoal"), "cap_closed"] += ret_vint(
                    bcoal_init, int(y + 1 - elec_data_run.loc[(y, "BrownCoal"), "Lifetime"])
                )

        else:
            #lock-in, potentially lifetime extension if profitable. Comparing variable cost of incumbent technologies with lcoe of alternative technologies
            df_elec = output_data.loc[(y,), "elec_gen"] / (
                output_data.loc[(y,), "cap_stock"]
                * elec_data_run.loc[(y,), "LF_max"]
                * 8760
                / 1000
            )
            print("df_elec", df_elec)
            df_elec[df_elec == 0] = 0.0001
            print("df_elec_updated", df_elec)
            
            op_cost = (
                elec_data_run.loc[y, "Fuel_cost_fuel"] / elec_data_run.loc[y, "Fuel_efficiency"]
                + elec_data_run.loc[y, "Variable_OM"]
            )/1000 + elec_data_run.loc[y, "Fixed_OM_annual"]/(24 * 365 * elec_data_run.loc[y, "LF_max"]*df_elec)
            
            
            share_extend = pd.Series(
                data=0, index=["Nuclear", "BrownCoal", "Gas", "HardCoal"]
            )
            for c in ["Nuclear","BrownCoal", "Gas", "HardCoal"]:  # delete nuclear

                if c in tech:
                    index_max_temp = np.unique(index_max).tolist()
                    if c in index_max_temp:
                        index_max_temp.remove(c)

                    corresp_lcoe = pd.Series(
                        data=alcoe(
                            elec_data_run.loc[y, "Inv"].values,
                            elec_data_run.loc[y, "Fixed_OM_annual"].values,
                            elec_data_run.loc[y, "Variable_OM"].values,
                            elec_data_run.loc[y, "Fuel_efficiency"].values,
                            elec_data_run.loc[y, "Fuel_cost_fuel"].values,
                            discount,
                            elec_data_run.loc[y, "Lifetime"].values,
                            elec_data_run.loc[y, "LF_max"].values,
                            df_elec.loc[c],
                        ),
                        index=tech,
                    )
                    corresp_lcoe.loc[c] = op_cost.loc[c]
                    corresp_lcoe[corresp_lcoe < 0] = 0
                    print("corresp_lcoe", corresp_lcoe)
                    
                    #lbda=30 --> we assume a high cost sensitivity for lifetime extension
                    
                    share_extend.loc[c] = share_inv(
                        c,
                        30,
                        corresp_lcoe,
                        np.concatenate((["Import", "Storage"], index_max_temp)),
                    )

                    print("share extend", c, share_extend)
                    if (df_elec.loc[c] < 0.01) or (
                        output_data.loc[(y, c), "cap_stock"] == 0
                    ):
                        # avoiding useless extension because of low df value leading to high lcoe for alternative tech
                        share_extend.loc[c] = 0



            output_data.loc[(y + 1, "Gas"), "cap_closed"] += ext_cap(
                gas_init,
                int(y + 1 - elec_data_run.loc[(y, "Gas"), "Lifetime"]),
                share_extend.loc["Gas"],
            )
            output_data.loc[(y + 1, "HardCoal"), "cap_closed"] += ext_cap(
                hcoal_init,
                int(y + 1 - elec_data_run.loc[(y, "HardCoal"), "Lifetime"]),
                share_extend.loc["HardCoal"],
            )
            output_data.loc[(y + 1, "Nuclear"), "cap_closed"] += ext_cap(
                nuke_init,
                int(y + 1 - elec_data_run.loc[(y, "Nuclear"), "Lifetime"]),
                share_extend.loc["Nuclear"],
            )
            if "BrownCoal" in tech:
                output_data.loc[(y + 1, "BrownCoal"), "cap_closed"] += ext_cap(
                    bcoal_init,
                    int(y + 1 - elec_data_run.loc[(y, "BrownCoal"), "Lifetime"]),
                    share_extend.loc["BrownCoal"],
                )

            # When lifetime extension decided in the model, the capacity lifetime added is 10 years            
            output_data.loc[(y + 1, "Gas"), "cap_closed"] += ret_vint(
                gas_init,
                int(y + 1 - elec_data_run.loc[(y, "Gas"), "Lifetime"]) - lifetime_ext,
            )
            output_data.loc[(y + 1, "HardCoal"), "cap_closed"] += ret_vint(
                hcoal_init,
                int(y + 1 - elec_data_run.loc[(y, "HardCoal"), "Lifetime"]) - lifetime_ext,
            )
            output_data.loc[(y + 1, "Nuclear"), "cap_closed"] += ret_vint(
                nuke_init,
                int(y + 1 - elec_data_run.loc[(y, "Nuclear"), "Lifetime"]) - lifetime_ext,
            )
            
            if "BrownCoal" in tech:
                output_data.loc[(y + 1, "BrownCoal"), "cap_closed"] += ret_vint(
                    bcoal_init,
                    int(y + 1 - elec_data_run.loc[(y, "BrownCoal"), "Lifetime"])
                    - lifetime_ext,
                )

    # Exogenous retirement (retirement of initial capacity observed) for other technologies
    for c in tech:
        if c not in ["Gas", "HardCoal", "BrownCoal", "Nuclear"]:
            output_data.loc[(y + 1, c), "cap_closed"] += elec_data_run.loc[
                (y + 1, c), "Initial_retired_capacity"
            ]

    # Endogeneous retirement for all capacities because of lifetime of capacity added on the 1990-2020 period
    output_data.loc[y + 1, "cap_closed"] = (
        output_data.loc[y + 1, "cap_closed"].values
        + output_data.loc[lifetime_index, "cap_new"].values
    )
    output_data.loc[lifetime_index, "cap_new"] = 0

    # #D.4 Capacity evolution
    output_data.loc[y + 1, "cap_stock"] = (
        output_data.loc[y, "cap_stock"].values
        + output_data.loc[y + 1, "cap_new"].values
        - output_data.loc[y + 1, "cap_closed"].values
    )
    output_data.loc[y + 1, "cap_new_inst"]=output_data.loc[y + 1, "cap_new"].values

    condition = ((output_data['cap_stock']<1e-08) & 
                (output_data.index.get_level_values(0)==(y+1)))
    output_data.loc[condition,'cap_stock']=0
    
    output_data.loc[output_data['cap_new']<1e-08,'cap_new']=0
    
 