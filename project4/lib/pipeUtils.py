from Levenshtein import ratio
import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin

#from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

'''
df.drop(columns=["Comments", "YearsENERGYSTARCertified"], inplace=True)
df.drop(columns=["DataYear"], inplace=True)
df.drop(columns=["Electricity(kWh)", "NaturalGas(therms)"], inplace=True)
'''

        
_df_ref = None

def numpy_nan_mean(a):
    return np.NaN if np.all(a!=a) else np.mean(a)


def init_df_ref(df):
    '''
        This function sets the reference dataframe for all defined functions. Therefore, it must be called at the beginning
        Parameters:
            df (DataFrame): reference dataframe
    '''
    global _df_ref
    _df_ref = df
    

def get_df_ref():
    global _df_ref
    
    return _df_ref


def _get_addr_ratio(x, v):
    return ratio(
        x.Address.lower().strip()+x.City.lower().strip()+x.ZipCode().strip(),
        v.Address.lower().strip()+v.City.lower().strip()+v.ZipCode().strip()
    )


def fix_missing_lat_lon_se(v):
    '''
        Fixes a single observation Latitude and Longitude from its address from its address, city and zipcode
        by looking for the closest one in initial df (Levehnstein ratio)
        Parameters:
            v (object, dictionary): observation to get Longitude and Latitude from
        Returns:
            Dictionnary with "Longitude" and "Latitude" keys
    '''
    global _df_ref
    if _df_ref is None:
        raise Exception("reference Dataframe has not been provided. Please provide it through init_df_ref function")

    if v.Address.lower().strip()+v.City.lower().strip()+v.ZipCode().strip()=="":
        return numpy_nan_mean(df.loc[:,["Longitude","Latitude"]])
    
    df_copy = _df_ref.loc[~df["Latitude"].isna() & ~_df_ref["Longitude"].isna(), ["Address", "City", "ZipCode", "Latitude", "Longitude"]]
    df_copy["score"] = df_copy.apply(lambda x: _get_addr_ratio(x, v), axis=0)
    return df_copy.sort_values("ratio").reset_index().loc[0, ["Longitude", "Latitude"]]
    

def fix_col_by_median_se(o, col_to_fix, ref_cols):
    '''
        Fixes a single observation element at col col_to_fix with median values of initial dataframe values at columns ref_cols
        Parameters:
            o (object, dictionary): observation to fix
            col_to_fix (str): column name to fix
            ref_cols (array): list of columns to with same value as v
        Returns:
            (object, dictionary): fixed observation
    '''
    global _df_ref
    if _df_ref is None:
        raise Exception("reference Dataframe has not been provided. Please provide it through init_df_ref function")

    df2 = _df_ref.copy()
    v = o.copy()
    for col in ref_cols:
        if pd.isna(v[col]):
            return v
        df2 = df2.loc[df2[col]==v[col]]
    if df2.shape[0]>0:
        v[col_to_fix] = numpy_nan_mean(df2[col_to_fix])
    return v
    
    
def fix_col_by_median(df,col_to_fix, ref_cols):
    '''
        Fixes all observations with Na value at col col_to_fix with median values of initial dataframe values at columns ref_cols
        Parameters:
            df (DataFrame): Dataframe to fix
            col_to_fix (str): column name to fix
            ref_cols (array): list of columns to with same value as v
        Returns:
            (object, dictionary): fixed observation
    '''
    ref_indexes = df.loc[df[col_to_fix].isna()].index
    for ref_index in ref_indexes:
        df.iloc[ref_index] = fix_col_by_median_se(
            df.iloc[ref_index],
            col_to_fix,
            ref_cols
        )
        
        
def get_zip_code_from_lat_lon(lat, lon):
    '''
        Returns guessed zip code from latitute and longitude
        Parameters:
            lat (float): Latitude
            lon (float): Longitude
        Returns:
            Zip code
    '''
    global _df_ref
    if _df_ref is None:
        raise Exception("reference Dataframe has not been provided. Please provide it through init_df_ref function")

    df_copy = _df_ref.loc[
        ~_df_ref["ZipCode"].isna() & ~_df_ref["Latitude"].isna() & ~_df_ref["Longitude"].isna(), 
        ["Latitude", "Longitude", "ZipCode"]
    ].copy()
    df_copy["distance"] = (df_copy["Latitude"] - lat)**2 + (df_copy["Longitude"] - lon)**2
    return df_copy.sort_values("distance").reset_index().at[0, "ZipCode"]
        

def get_geo_col_from_lat_lon(col_name, lat, lon):
    '''
        Returns guessed column value from latitute and longitude
        Parameters:
            col_name(string): column to impute
            lat (float): Latitude
            lon (float): Longitude
        Returns:
            col value
    '''
    global _df_ref
    if _df_ref is None:
        raise Exception("reference Dataframe has not been provided. Please provide it through init_df_ref function")

    df_copy = _df_ref.loc[
        ~_df_ref[col_name].isna() & ~_df_ref["Latitude"].isna() & ~_df_ref["Longitude"].isna(), 
        ["Latitude", "Longitude", col_name]
    ].copy()
    df_copy["distance"] = (df_copy["Latitude"] - lat)**2 + (df_copy["Longitude"] - lon)**2
    return df_copy.sort_values("distance").reset_index().at[0, col_name]
        

def fix_number_of_floors_se(o):
    '''
        Fixes number of floors (NumberofFloors) for a single observation o
        Parameters:
            o (object, dictionary): observation to fix
        Returns:
            (object, dictionary): fixed observation
    '''
    global _df_ref
    if _df_ref is None:
        raise Exception("reference Dataframe has not been provided. Please provide it through init_df_ref function")

    v = fix_col_by_median_se(o, "NumberofFloors", ["BuildingType", "PrimaryPropertyType", "Neighborhood", "YearBuilt"])
    if (pd.isna(v["NumberofFloors"])):
        v = fix_col_by_median_se(v, "NumberofFloors", ["BuildingType", "PrimaryPropertyType", "YearBuilt"])
    if (pd.isna(v["NumberofFloors"])):
        v = fix_col_by_median_se(v, "NumberofFloors", ["BuildingType", "PrimaryPropertyType"])
    return v


def fix_number_of_floors(df):
    '''
        Fixes all observations without NumberofFloors (0 or Nan)
    '''
    global _df_ref
    df.loc[df["NumberofFloors"]==0, "NumberofFloors"] = np.NaN;
    for ref_index in df.loc[df["NumberofFloors"].isna()].index:
        df.iloc[ref_index] = fix_number_of_floors_se(df.iloc[ref_index])

    
class DataFrameImputer(TransformerMixin):

    def __init__(self, ref_cols, col_to_fix, method="median"):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.
        Columns of other types are imputed with mean or median depending on the method parameter of column.

        """
        self.method = method
        self.col_to_fix = col_to_fix
        self.ref_cols = ref_cols
        
    def fit(self, X):
        self.fill = X
        return self

    def transform(self, X):
        for ind in X.loc[X[self.col_to_fix].isna()].index:
            X_filtered = X.copy()

            X_filtered = X_filtered.loc[~X_filtered[self.col_to_fix].isna()]
            for c in self.ref_cols:
                v = X.at[ind, c]
                X_filtered = X_filtered.loc[~X_filtered[c].isna() & (X_filtered[c]==v)]
                
            if X_filtered.shape[0]==0:
                continue
            if X[c].dtype == np.dtype('O'):
                X.at[ind, self.col_to_fix] = X_filtered[self.col_to_fix].value_counts().index[0]
            else:
                if self.method=="mean":
                    X.at[ind, self.col_to_fix] = X_filtered[self.col_to_fix].mean()
                else:
                    X.at[ind, self.col_to_fix] = X_filtered[self.col_to_fix].median()
        return X
        
def clean_and_impute(df_orig, inplace=False):
    '''
        Cleans and imputes a dataframe from Seattle xxx
        Parameters:
            df_orig (DataFrame): Dataframe to clean and impute
        Returns:
            DataFrame: cleaned and imputed 
    '''

    if inplace==False:
        df = df_orig.copy(deep=True)
    else:
        df = df_orig
        
    
    # Filtrage des immeubles non destinés à l'habitation
    df = df.loc[~df["BuildingType"].str.lower().str.contains("family")].reset_index(drop=True)
    df = df.loc[~df["PrimaryPropertyType"].str.lower().str.contains("family")].reset_index(drop=True)
    
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].str.upper()

    
    df.loc[df["BuildingType"]=="NONRESIDENTIAL WA", "BuildingType"] = "NONRESIDENTIAL"
    df.loc[df["BuildingType"]=="NONRESIDENTIAL COS", "BuildingType"] = "NONRESIDENTIAL"

    # Nettoyage des quartiers
    df["Neighborhood"] = df["Neighborhood"].str.upper()
    df.loc[df["Neighborhood"]=="DELRIDGE NEIGHBORHOODS", "Neighborhood"] = "DELRIDGE"
    
    # Imputation des Second et Third largestPropertyUseType par 0 et ""
    df.loc[df["SecondLargestPropertyUseTypeGFA"].isna(), "SecondLargestPropertyUseTypeGFA"] = 0
    df.loc[df["ThirdLargestPropertyUseTypeGFA"].isna(), "ThirdLargestPropertyUseTypeGFA"] = 0
    df.loc[df["SecondLargestPropertyUseType"].isna(), "SecondLargestPropertyUseType"] = ""
    df.loc[df["ThirdLargestPropertyUseType"].isna(), "ThirdLargestPropertyUseType"] = ""
    
    # Suppression de colonnes non remplies
    df.drop(columns=["Comments", "YearsENERGYSTARCertified"], inplace=True)

    init_df_ref(df)
    
    # Imputation de LargestPropertyUseType
    df = DataFrameImputer(ref_cols=["PrimaryPropertyType"], col_to_fix="LargestPropertyUseType").fit_transform(df)
    
    # Imputation de LargestPropertyUseTypeGFA
    df = DataFrameImputer(ref_cols=["PrimaryPropertyType", "BuildingType", "Neighborhood"], col_to_fix="LargestPropertyUseTypeGFA").fit_transform(df)
    df = DataFrameImputer(ref_cols=["PrimaryPropertyType", "Neighborhood"], col_to_fix="LargestPropertyUseTypeGFA").fit_transform(df)
    df = DataFrameImputer(ref_cols=["PrimaryPropertyType", "BuildingType"], col_to_fix="LargestPropertyUseTypeGFA").fit_transform(df)
    df = DataFrameImputer(ref_cols=["PrimaryPropertyType"], col_to_fix="LargestPropertyUseTypeGFA").fit_transform(df)
    
    
    # Imputation de PropertyGFABuilding(s)
    df = DataFrameImputer(ref_cols=["PrimaryPropertyType", "BuildingType", "Neighborhood"], col_to_fix="PropertyGFABuilding(s)").fit_transform(df)
    df = DataFrameImputer(ref_cols=["PrimaryPropertyType", "Neighborhood"], col_to_fix="PropertyGFABuilding(s)").fit_transform(df)
    df = DataFrameImputer(ref_cols=["PrimaryPropertyType", "BuildingType"], col_to_fix="PropertyGFABuilding(s)").fit_transform(df)
    df = DataFrameImputer(ref_cols=["PrimaryPropertyType"], col_to_fix="PropertyGFABuilding(s)").fit_transform(df)
    
    # Correction des surfaces incohérentes
    for ref_index in df.loc[df["PropertyGFATotal"] < (df["LargestPropertyUseTypeGFA"] + df["SecondLargestPropertyUseTypeGFA"] + df["ThirdLargestPropertyUseTypeGFA"])].index:
        total = df.at[ref_index,"LargestPropertyUseTypeGFA"] + df.at[ref_index,"SecondLargestPropertyUseTypeGFA"] + df.at[ref_index,"ThirdLargestPropertyUseTypeGFA"]
        diff = total - df.at[ref_index, "PropertyGFATotal"]
        df.at[ref_index, "PropertyGFATotal"] = total 
        df.at[ref_index, "PropertyGFABuilding(s)"] = df.at[ref_index, "PropertyGFABuilding(s)"] + diff

    # Transformation de YearBuilt en DecadeBuilt
    df["DecadeBuilt"] = np.char.mod("%d", np.rint(df["YearBuilt"]/5) * 5)
    df.drop(columns=["YearBuilt"], inplace=True)


    # Imputation de NumberofBuildings par voisinage
    df = DataFrameImputer(ref_cols=["PrimaryPropertyType", "BuildingType", "Neighborhood", "DecadeBuilt"], col_to_fix="NumberofBuildings").fit_transform(df)
    df = DataFrameImputer(ref_cols=["PrimaryPropertyType", "Neighborhood", "DecadeBuilt"], col_to_fix="NumberofBuildings").fit_transform(df)
    df = DataFrameImputer(ref_cols=["PrimaryPropertyType", "BuildingType", "DecadeBuilt"], col_to_fix="NumberofBuildings").fit_transform(df)
    df = DataFrameImputer(ref_cols=["PrimaryPropertyType", "DecadeBuilt"], col_to_fix="NumberofBuildings").fit_transform(df)
    df = DataFrameImputer(ref_cols=["PrimaryPropertyType"], col_to_fix="NumberofBuildings").fit_transform(df)

    # Correction de NumberofFloors : le plus grand immeuble compte 76 étages
    df.loc[df["NumberofFloors"]>76, "NumberofFloors"] = np.nan
    df.loc[df["NumberofFloors"]<0, "NumberofFloors"] = np.nan
    df = DataFrameImputer(ref_cols=["PrimaryPropertyType", "BuildingType", "Neighborhood", "DecadeBuilt"], col_to_fix="NumberofFloors").fit_transform(df)
    df = DataFrameImputer(ref_cols=["PrimaryPropertyType", "Neighborhood", "DecadeBuilt"], col_to_fix="NumberofFloors").fit_transform(df)
    df = DataFrameImputer(ref_cols=["PrimaryPropertyType", "BuildingType", "DecadeBuilt"], col_to_fix="NumberofFloors").fit_transform(df)
    df = DataFrameImputer(ref_cols=["PrimaryPropertyType", "DecadeBuilt"], col_to_fix="NumberofFloors").fit_transform(df)
    df = DataFrameImputer(ref_cols=["PrimaryPropertyType"], col_to_fix="NumberofFloors").fit_transform(df)

    
    # Correction des PropertyGFAParking
    df["parking_ratio"] = df["PropertyGFAParking"] / df["PropertyGFABuilding(s)"]
    max_ratio_paking = 4
    #init_df_ref(df)
    df.loc[df["parking_ratio"]>max_ratio_paking, "parking_ratio"] = np.nan
    df = DataFrameImputer(ref_cols=["PrimaryPropertyType", "BuildingType", "Neighborhood", "DecadeBuilt"], col_to_fix="parking_ratio").fit_transform(df)
    df = DataFrameImputer(ref_cols=["PrimaryPropertyType", "BuildingType", "DecadeBuilt"], col_to_fix="parking_ratio").fit_transform(df)
    df = DataFrameImputer(ref_cols=["Neighborhood", "BuildingType", "DecadeBuilt"], col_to_fix="parking_ratio").fit_transform(df)
    df = DataFrameImputer(ref_cols=["Neighborhood", "BuildingType"], col_to_fix="parking_ratio").fit_transform(df)
    df["PropertyGFAParking"] = df["PropertyGFABuilding(s)"] * df["parking_ratio"]

    # Récupération de l' électricity et du gaz manquants en kBtu à partir des autres unités
    df.loc[df["Electricity(kBtu)"].isna() & ~df["Electricity(kWh)"].isna(), "Electricity(kBtu)"] = df["Electricity(kWh)"] * 3.413
    df.loc[df["NaturalGas(kBtu)"].isna() & ~df["NaturalGas(therms)"].isna(), "NaturalGas(kBtu)"] = df["NaturalGas(therms)"] * 99.976
    
    # Suppression des colonnes dupliquées dans des unités différentes de kBtu
    df.drop(columns=["Electricity(kWh)", "NaturalGas(therms)"], inplace=True)

    # Très peu d' immeubles fonctionnent avec la vapeur
    df.loc[df["SteamUse(kBtu)"].isna(), "SteamUse(kBtu)"] = 0

    # Les énergies consommées en gaz et en électricité ne peuvent pas être négatives
    df.loc[df["NaturalGas(kBtu)"]<0, "NaturalGas(kBtu)"] = np.nan
    df.loc[df["Electricity(kBtu)"]<0, "Electricity(kBtu)"] = np.nan
    df.loc[df["SiteEnergyUse(kBtu)"]<=0, "SiteEnergyUse(kBtu)"] = np.nan

    # Imputation de SiteEnergyUse(kBtu) par somme des énergies
    df.loc[df["SiteEnergyUse(kBtu)"].isna(), "SiteEnergyUse(kBtu)"] = df["NaturalGas(kBtu)"] + df["Electricity(kBtu)"] + df["SteamUse(kBtu)"]
                                                                         
        
    for col_target in ["SiteEUIWN(kBtu/sf)", "GHGEmissionsIntensity"]:
        df.loc[df[col_target]<0, col_target] = np.nan
        df = DataFrameImputer(ref_cols=["PrimaryPropertyType", "BuildingType", "Neighborhood", "DecadeBuilt"], col_to_fix=col_target).fit_transform(df)
        df = DataFrameImputer(ref_cols=["PrimaryPropertyType", "BuildingType", "DecadeBuilt"], col_to_fix=col_target).fit_transform(df)
        df = DataFrameImputer(ref_cols=["Neighborhood", "LargestPropertyTypeUse", "DecadeBuilt"], col_to_fix=col_target).fit_transform(df)
        df = DataFrameImputer(ref_cols=["Neighborhood", "LargestPropertyTypeUse"], col_to_fix=col_target).fit_transform(df)
        df = DataFrameImputer(ref_cols=["LargestPropertyTypeUse", "DecadeBuilt"], col_to_fix=col_target).fit_transform(df)
        df = DataFrameImputer(ref_cols=["DecadeBuilt"], col_to_fix=col_target).fit_transform(df)
        df = DataFrameImputer(ref_cols=["LargestPropertyTypeUse"], col_to_fix=col_target).fit_transform(df)
        df = DataFrameImputer(ref_cols=["BuildingType"], col_to_fix=col_target).fit_transform(df)

    df["gas_ratio"] = df["NaturalGas(kBtu)"]/df["SiteEnergyUse(kBtu)"]
    df["electricity_ratio"] = df["Electricity(kBtu)"]/df["SiteEnergyUse(kBtu)"]
    df["steam_ratio"] = df["SteamUse(kBtu)"]/df["SiteEnergyUse(kBtu)"]
    
    df["largest_property_use_type_ratio"] = np.divide(df["LargestPropertyUseTypeGFA"], df["PropertyGFATotal"])
    df["second_largest_property_use_type_ratio"] = np.divide(df["SecondLargestPropertyUseTypeGFA"], df["PropertyGFATotal"])
    df["third_largest_property_use_type_ratio"] = np.divide(df["ThirdLargestPropertyUseTypeGFA"], df["PropertyGFATotal"])
    
    for col_target in ["gas_ratio", "electricity_ratio", "steam_ratio"]:
        df = DataFrameImputer(ref_cols=["PrimaryPropertyType", "BuildingType", "Neighborhood", "DecadeBuilt"], col_to_fix=col_target).fit_transform(df)
        df = DataFrameImputer(ref_cols=["PrimaryPropertyType", "BuildingType", "DecadeBuilt"], col_to_fix=col_target).fit_transform(df)
        df = DataFrameImputer(ref_cols=["Neighborhood", "BuildingType", "DecadeBuilt"], col_to_fix=col_target).fit_transform(df)
        df = DataFrameImputer(ref_cols=["Neighborhood", "BuildingType"], col_to_fix=col_target).fit_transform(df)
        df = DataFrameImputer(ref_cols=["PrimaryPropertyType", "DecadeBuilt"], col_to_fix=col_target).fit_transform(df)
        df = DataFrameImputer(ref_cols=["DecadeBuilt"], col_to_fix=col_target).fit_transform(df)
        df = DataFrameImputer(ref_cols=["PrimaryPropertyType"], col_to_fix=col_target).fit_transform(df)
        df = DataFrameImputer(ref_cols=["BuildingType"], col_to_fix=col_target).fit_transform(df)

    
    # Suppression des colonnes inutiles
    df.drop(columns=[
        "Outlier", "ComplianceStatus", "DefaultData", "NaturalGas(kBtu)", "Electricity(kBtu)", "SteamUse(kBtu)", "SiteEnergyUseWN(kBtu)",
        "SiteEnergyUseWN(kBtu)", "SourceEUIWN(kBtu/sf)", "SourceEUI(kBtu/sf)", "SiteEUI(kBtu/sf)", "Longitude", "Latitude", "OSEBuildingID", "DataYear",
        "PropertyName", "Address", "City", "State", "ZipCode", "TaxParcelIdentificationNumber", "CouncilDistrictCode", "SiteEnergyUse(kBtu)",
        "TotalGHGEmissions", "ListOfAllPropertyUseTypes", "PropertyGFAParking", 
        "LargestPropertyUseTypeGFA", "SecondLargestPropertyUseTypeGFA", "ThirdLargestPropertyUseTypeGFA"], inplace=True)
    return df
