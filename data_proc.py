#Code author: Vivien Fisch-Romito, University of Geneva
#Conceptualization : Vivien Fisch-Romito, Evelina Trutnevyte, University of Geneva
#Date of modification: 14/12/23
#


# import packages & library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def imp_socapt(filepath):
    """Importing and formatting social acceptance data
    I: filepath ; O: social acceptance dataframe"""
    soc_tech_data = pd.read_excel(filepath, header=[0], index_col="Country")

    soc_tech = (
        soc_tech_data.set_index("Year", append=True)
        .reindex(
            pd.MultiIndex.from_product(
                [
                    soc_tech_data.index.unique(),
                    range(soc_tech_data.Year.min(), soc_tech_data.Year.max() + 1),
                ],
                names=["Country", "Year"],
            )
        )
        .reset_index(level=1)
    )

    soc_tech = (
        soc_tech.groupby(level=[0])
        .apply(
            lambda g: g.reset_index(level=[0], drop=True).interpolate(method="index")
        )
        .reset_index()
        .drop(columns="level_1")
        .set_index(["Country", "Year"])
    )
    soc_tech = soc_tech.squeeze()
    return soc_tech


country_code = pd.read_excel(
    "./inputs/_common/Datasets/ISO_CountryCodes.xlsx", header=[0]
)

# Importing and formatting Global Coal plant tracker dataset for coal capacity vintage
gcpt = pd.read_excel(
    "./inputs/_common/Datasets/GlobalEnergyMonitor_GlobalCoalPlantTracker.xlsx",
    header=[0],
    usecols=[
        "Country",
        "Capacity (MW)",
        "Status",
        "Year",
        "RETIRED",
        "Coal type",
        "Region",
    ],
    index_col=None,
    sheet_name="Units",
)
gcpt = gcpt[gcpt["Region"].isin(["non-EU Europe", "EU27"])]
gcpt = gcpt[
    gcpt["Status"].isin(["Operating", "Retired"])
]  # Mothballed plant not included to be consistent with input dataset 

gcpt.loc[
    (gcpt["Coal type"].str.lower().str.contains("lignite"))
    | (gcpt["Coal type"].str.lower().str.contains("peat"))
    | (gcpt["Coal type"].str.lower().str.contains("shale")),
    "Coal type",
] = "BrownCoal"
gcpt.loc[gcpt["Coal type"] != "BrownCoal", "Coal type"] = "HardCoal"

index_drop = gcpt[
    (gcpt["Status"] == "Retired")
    & ((gcpt["RETIRED"] < 1991) | (gcpt["RETIRED"].isna()))
].index
gcpt.drop(index_drop, inplace=True)
gcpt.drop(
    gcpt[(gcpt["Year"] > 1990) | (gcpt["Year"].isna())].index, inplace=True
)  

gcpt["Year"] = gcpt["Year"].astype(str)
gcpt = gcpt.merge(country_code, how="left")
gcpt = (
    gcpt[["Alpha-3 code", "Capacity (MW)", "Year", "Coal type"]]
    .groupby(["Alpha-3 code", "Year", "Coal type"])
    .sum()
)
gcpt = gcpt.reset_index()
gcpt["Year"] = gcpt["Year"].astype(int)
gcpt.set_index(["Alpha-3 code", "Year", "Coal type"], inplace=True)
gcpt = gcpt.squeeze() / 1000


average_year_hc = int(
    np.mean(gcpt.loc[:, :, "HardCoal"].index.get_level_values(1)).round()
)


average_year_bc = int(
    np.mean(gcpt.loc[:, :, "BrownCoal"].index.get_level_values(1)).round()
)


# Importing and formatting global Gas Plant Tracker dataset for gas capacity vintage
ggpt = pd.read_excel(
    "./inputs/_common/Datasets/GlobalEnergyMonitor_GlobalGasPlantTracker.xlsx",
    header=[0],
    usecols=["Region", "Country", "Capacity (MW)", "Status", "Year", "RETIRED"],
    index_col=None,
    sheet_name="Units",
)
ggpt = ggpt[ggpt["Region"].isin(["non-EU Europe", "Europe"])]
ggpt = ggpt[
    ggpt["Status"].isin(["Operating", "Retired"])
]  # Mothballed plant not included to be consistent with input dataset 
index_drop = ggpt[
    (ggpt["Status"] == "Retired")
    & ((ggpt["RETIRED"] < 1991) | (ggpt["RETIRED"].isna()))
].index
ggpt.drop(index_drop, inplace=True)

for i in ggpt.index:
    if "-" in str(ggpt.loc[i, "Year"]):
        ggpt.loc[i, "Year"] = ggpt.loc[i, "Year"][0:4]
    if "," in str(ggpt.loc[i, "Year"]):
        ggpt.loc[i, "Year"] = ggpt.loc[i, "Year"][0:4]

mean_start = ggpt.copy()
mean_start = mean_start[
    (mean_start["Year"] != "not found") & (mean_start["Status"] == "Operating")
]
mean_start["Year"] = pd.to_numeric(mean_start["Year"])
mean_start = (
    mean_start[["Country", "Year"]]
    .groupby("Country")
    .mean()
    .round()
    .squeeze()
    .astype(int)
)

for c in ggpt.loc[
    (ggpt["Year"] == "not found") & (ggpt["Status"] == "Operating"), "Country"
].unique():
    ggpt.loc[
        (ggpt["Year"] == "not found")
        & (ggpt["Status"] == "Operating")
        & (ggpt["Country"] == c),
        "Year",
    ] = mean_start.loc[c]

ggpt.drop(ggpt[pd.to_numeric(ggpt["Year"]) > 1990].index, inplace=True)
ggpt["Year"] = ggpt["Year"].astype(str)
ggpt = ggpt.merge(country_code, how="left")
ggpt = (
    ggpt[["Alpha-3 code", "Year", "Capacity (MW)"]]
    .groupby(["Alpha-3 code", "Year"])
    .sum()
)
ggpt = ggpt.reset_index()
ggpt["Year"] = ggpt["Year"].astype(int)
ggpt.set_index(["Alpha-3 code", "Year"], inplace=True)
ggpt = ggpt.squeeze() / 1000

average_year_g = int(np.mean(ggpt.index.get_level_values(1)).round())


# Importing and formatting data from JRC for nuclear capacity vintage
jrc_nuclear = pd.read_csv(
    "./inputs/_common/Datasets/JRC_OPEN_UNITS.csv",
    usecols=["capacity_g", "type_g", "country", "status_g", "year_commissioned"],
    index_col=None,
)
jrc_nuclear = jrc_nuclear.loc[jrc_nuclear["type_g"] == "Nuclear"]

mean_start = jrc_nuclear.copy()
mean_start = mean_start[(mean_start["year_commissioned"].notna())]
mean_start["year_commissioned"] = pd.to_numeric(mean_start["year_commissioned"])
mean_start = (
    mean_start[["country", "year_commissioned"]]
    .groupby("country")
    .mean()
    .round()
    .squeeze()
    .astype(int)
)

for c in jrc_nuclear.loc[(jrc_nuclear["year_commissioned"].isna()), "country"].unique():
    jrc_nuclear.loc[
        (jrc_nuclear["year_commissioned"].isna()) & (jrc_nuclear["country"] == c),
        "year_commissioned",
    ] = mean_start.loc[c]

jrc_nuclear.drop(
    jrc_nuclear[pd.to_numeric(jrc_nuclear["year_commissioned"] > 1990)].index,
    inplace=True,
)
jrc_nuclear["year_commissioned"] = (
    jrc_nuclear["year_commissioned"].astype(int).astype(str)
)
jrc_nuclear.replace("Czechia", "Czech Republic", inplace=True)
jrc_nuclear.rename(columns={"country": "Country"}, inplace=True)
jrc_nuclear = jrc_nuclear.merge(country_code, how="left")
jrc_nuclear = (
    jrc_nuclear[["Alpha-3 code", "year_commissioned", "capacity_g"]]
    .groupby(["Alpha-3 code", "year_commissioned"])
    .sum()
)
jrc_nuclear = jrc_nuclear.reset_index()
jrc_nuclear["year_commissioned"] = jrc_nuclear["year_commissioned"].astype(int)
jrc_nuclear.set_index(["Alpha-3 code", "year_commissioned"], inplace=True)
jrc_nuclear = jrc_nuclear.squeeze() / 1000

average_year_nuke = int(np.mean(jrc_nuclear.index.get_level_values(1)).round())



# Importing and formatting data of public acceptance for technologies
soc_nuc = imp_socapt("./inputs/_common/Datasets/Eurobarometer_nuclear.xlsx")
soc_wind = imp_socapt("./inputs/_common/Datasets/Eurobarometer_wind.xlsx")
soc_coal = imp_socapt("./inputs/_common/Datasets/Eurobarometer_coal.xlsx")
soc_gas = imp_socapt("./inputs/_common/Datasets/Eurobarometer_gas.xlsx")
soc_pv = imp_socapt("./inputs/_common/Datasets/Eurobarometer_pv.xlsx")

# Importing governance indicators
oecd_gov = pd.read_excel(
    "./inputs/_common/Datasets/OECD_governance_indicators.xlsx",
    header=[0],
    index_col=[0, 1],
)

# Importing Weighted average costs of capital (WACC)
wacc = (
        pd.read_excel(
            "./inputs/_common/Datasets/WACC_tech.xlsx", header=[0], index_col=[0,1]
        )
        .drop("Source", axis="columns")
    )
years=np.arange(2009,2019).tolist()
years.remove(2015)
for i in years:
    for c in ['PV', 'OnshoreWind', 'OffshoreWind', 'Biomass']:
        wacc.loc[pd.IndexSlice[:,i],c]=(wacc.loc[pd.IndexSlice[:,2015],c]-wacc.loc[pd.IndexSlice[:,2015],'Risk-free rate']).values+wacc.loc[pd.IndexSlice[:,i],'Risk-free rate'].values
wacc.loc[pd.IndexSlice[:,2019],:]=wacc.loc[pd.IndexSlice[:,2018],:].values
wacc=wacc.drop('Risk-free rate', axis='columns')
