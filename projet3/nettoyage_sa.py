#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd
import numpy as np
import lib.cleanUtils as cu
import math
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import itertools
import sys
import csv

def product_in_country(x, terms):
    if pd.isna(x["countries_tags"]):
        return False

    for t in terms:
        if str(t) in x["countries_tags"].lower():
            return True

    return False

def fix_out_of_bounds_values(df, col_to_fix="", lowest_limit=0, highest_limit=100):
    if col_to_fix.strip() == "":
        raise Exception("Aucune colonne fournie")
    if col_to_fix not in df.columns:
        raise (f"La colonne {col_to_fix} n'existe pas dans le dataframe fourni.")

    for _ref_col in [
        "product_name",
        "main_category",
        "food_groups_tags",
        "pnns_groups_2",
        "pnns_groups_1",
    ]:
        if (
            df.loc[
                (df[col_to_fix] < lowest_limit) | (df[col_to_fix] > highest_limit)
            ].shape[0]
            > 0
        ):
            cu.fix_outliers_from_ref_column_equality(
                df,
                ref_col=_ref_col,
                col_to_fix=col_to_fix,
                lower_lim=lowest_limit,
                upper_lim=highest_limit,
                verbose=False,
                show_progress=False,
            )
    if (
        df.loc[
            (df[col_to_fix] < lowest_limit) | (df[col_to_fix] > highest_limit)
        ].shape[0]
        > 0
    ):
        for ind in df.loc[
            (df[col_to_fix] < lowest_limit) | (df[col_to_fix] > highest_limit)
        ].index:
            cu.fix_row_col_with_median_most_important_words(
                df, "product_name", ind, col_to_fix, imp_p_n_terms
            )

def fix_errors_values(df, index_to_fix, col_to_fix=""):
    if col_to_fix.strip() == "":
        raise Exception("Aucune colonne fournie")
    if col_to_fix not in df.columns:
        raise (f"La colonne {col_to_fix} n'existe pas dans le dataframe fourni.")

    for _ref_col in [
        "product_name",
        "main_category",
        "food_groups_tags",
        "pnns_groups_2"
    ]:
        ref_val = df.loc[index_to_fix, _ref_col]
        if pd.isna(ref_val):
            continue
        ref_indexes = df.loc[df[_ref_col]==ref_val].index
        cu.fix_var_with_median_ref_indexes(
            df, ref_indexes, index_to_fix, col_to_fix, verbose=False
        )

def get_pnns1_from_pnns2(pnns2):
    for pnns1 in list(pnns_groups.keys()):
        if pnns2 in pnns_groups[pnns1]:
            return pnns1
    return None

def fill_pnns_1(x):
    if pd.isna(x["pnns_groups_1"]) and pd.isna(x["pnns_groups_2"])==False:
        return get_pnns1_from_pnns2(x["pnns_groups_2"])
    else:
        return x["pnns_groups_1"]


def clean_data(input_filename, output_filename):
    working_dir = "working_stand_alone"

    if not os.path.exists(working_dir):
        os.mkdir(working_dir)


    df = pd.read_csv(input_filename, sep="\t", nrows=500000)
    types = dict(zip(df.columns, df.dtypes))

    dtypes["additives"] = "str"
    dtypes["cities"] = "str"
    other_cols_to_del = []

    for i in range(len(df.columns) - 1, -1, -1):
        if df.columns[i].endswith("url"):
            other_cols_to_del.append(df.columns[i])

    for i in range(len(df.columns) - 1, -1, -1):
        if df.columns[i].endswith("datetime") or df.columns[i].endswith("_t"):
            other_cols_to_del.append(df.columns[i])

    df.loc[
        ~df["packaging"].isna(),
        ["packaging", "packaging_tags", "packaging_en", "packaging_text"],
    ]

    other_cols_to_del = np.concatenate(
        (other_cols_to_del, ["packaging", "packaging_en", "packaging_text"])
    )


    other_cols_to_del = other_cols_to_del = np.append(other_cols_to_del, "labels")
    other_cols_to_del = np.append(other_cols_to_del, "labels_en")
    other_cols_to_del = np.append(other_cols_to_del, "manufacturing_places")
    other_cols_to_del = np.append(other_cols_to_del, "emb_codes")
    other_cols_to_del = np.append(other_cols_to_del, "countries")
    other_cols_to_del = np.append(other_cols_to_del, "countries_en")
    other_cols_to_del = np.append(other_cols_to_del, "ingredients_text")
    other_cols_to_del = np.append(other_cols_to_del, "traces")
    other_cols_to_del = np.append(other_cols_to_del, "traces_en")
    other_cols_to_del = np.append(other_cols_to_del, "additives_en")
    other_cols_to_del = np.append(other_cols_to_del, "food_groups")
    other_cols_to_del = np.append(other_cols_to_del, "food_groups_en")
    other_cols_to_del = np.append(other_cols_to_del, "states")
    other_cols_to_del = np.append(other_cols_to_del, "states_en")
    other_cols_to_del = np.append(other_cols_to_del, "cities")
    other_cols_to_del = np.append(other_cols_to_del, "brands")
    other_cols_to_del = np.append(other_cols_to_del, "origins")
    other_cols_to_del = np.append(other_cols_to_del, "origins_en")
    other_cols_to_del = np.append(other_cols_to_del, "categories")
    other_cols_to_del = np.append(other_cols_to_del, "categories_en")


    df["countries_tags"].unique()
    skip_rows = 0
    nrows = 300000

    all_countries = {}
    ind = 1
    parquet_file_indexes = []
    repartition = dict(zip(df.columns, [0] * len(df.columns)))
    total_count = 0
    while True:
        df = pd.read_csv(
            "assets/en.openfoodfacts.org.products.csv",
            sep="\t",
            dtype=dtypes,
            nrows=nrows,
            skiprows=range(1, total_count),
        )

        if df.shape[0] == 0:
            break
        df.drop(columns=other_cols_to_del, inplace=True)
        parquet_fname = f"{working_dir}/en.openfoodfacts.org.products_{ind}.parquet"
        df.to_parquet(parquet_fname, engine="fastparquet")
        parquet_file_indexes.append(ind)
        countries = cu.extract_nb_terms_from_col(
            df, "countries_tags", sep=[","], min_length=0
        )
        for c2 in countries:
            if not c2 in all_countries:
                all_countries[c2] = 0
            all_countries[c2] += countries[c2]

        for col in df.columns:
            repartition[col] += df.loc[~df[col].isna()].shape[0]
        total_count += df.shape[0]
        ind += 1


    df_info = pd.DataFrame(
        {"column": repartition.keys(), "filled_nb": repartition.values()}
    )
    df_info["filled_ratio"] = round(100 * df_info["filled_nb"] / total_count, 2)
    df_info["dtype"] = df_info.apply(lambda x: dtypes[x["column"]], axis=1)
    df_info.to_parquet(f"{working_dir}/repartition_tous_pays.parquet")

    for c in all_countries:
        if "fren" in c or "fran" in c:
            print(c)

    from_france_terms = [
        "en:france",
        "en:french-polynesia",
        "en:french-guiana",
        "en:francia",
        "en:frankreich",
        "en:france-francais",
        "en:polynesie-francaise",
        "fr:francia",
        "en:frankrijk",
        "fr:frankreich",
        "fr:polinesia-francesa",
        "fr:francja",
        "en:polinesia-francesa",
        "fr:franța",
        "fr:france🇨🇵🇫🇷",
        "fr:frankrijk",
        "fr:fabrique-en-france",
        "en:francja",
        "fr:paris-france",
        "fr:france🇨🇵",
        "fr:france-only",
        "en:union-europeenne-france",
        "en:franța",
        "fr:franca",
        "en:franca",
        "es:franca",
        "en:frankrig",
        "ca:franca",
        "en:francuzka",
        "en:frankrike",
        "fr:france-🇫🇷",
        "de:frankreic",
    ]

    repartition_fr = dict(zip(df.columns, [0] * len(df.columns)))
    total_count_fr = 0
    for i in parquet_file_indexes:
        df = pd.read_parquet(f"{working_dir}/en.openfoodfacts.org.products_{i}.parquet")
        print(f"index: {i} before: {df.shape[0]}")
        df["keep"] = df.apply(lambda x: product_in_country(x, from_france_terms), axis=1)
        df.drop(df.loc[df["keep"] == False].index, inplace=True)
        df.drop(columns=["keep"], inplace=True)
        for col in df.columns:
            repartition_fr[col] += df.loc[~df[col].isna()].shape[0]
        total_count_fr += df.shape[0]

        print(f"index: {i} after: {df.shape[0]}")
        df.to_parquet(f"{working_dir}/in_france_products_{i}.parquet")

    df_info_fr = pd.DataFrame(
        {"column": repartition_fr.keys(), "filled_nb": repartition_fr.values()}
    )
    df_info_fr["filled_ratio"] = round(100 * df_info_fr["filled_nb"] / total_count_fr, 2)
    df_info_fr["dtype"] = df_info_fr.apply(lambda x: dtypes[x["column"]], axis=1)
    df_info_fr.to_parquet(f"{working_dir}/repartition_france.parquet")

    keep_columns = [
        "pnns_groups_1",
        "pnns_groups_2",
        "main_category",
        "food_groups_tags",
        "code",
        "product_name",
        "brands_tags",
        "nutriscore_score",
        "nutriscore_grade",
    ]
    for c in df_info_fr.loc[
        ((df_info_fr["dtype"] == "int64") | (df_info_fr["dtype"] == "float64"))
        & (df_info_fr["filled_ratio"] >= 50)
    ]["column"]:
        keep_columns.append(c)

    drop_columns = []
    for i in range(0, len(df.columns)):
        if not df.columns[i] in keep_columns:
            drop_columns.append(df.columns[i])

    for i in parquet_file_indexes:
        df = pd.read_parquet(f"{working_dir}/in_france_products_{i}.parquet")
        df.drop(columns=drop_columns, inplace=True)
        df.to_parquet(f"{working_dir}/in_france_trimmed_products_{i}.parquet")

    df = pd.read_parquet(f"{working_dir}/in_france_trimmed_products_1.parquet")
    for i in parquet_file_indexes:
        if i == 1:
            continue
        df2 = pd.read_parquet(f"{working_dir}/in_france_trimmed_products_{i}.parquet")
        df = pd.concat([df, df2])

    df.reset_index(drop=True, inplace=True)

    df.loc[df["pnns_groups_1"].str.lower().str.strip()=="unknown", "pnns_groups_1"] = None
    df.loc[df["pnns_groups_2"].str.lower().str.strip()=="unknown", "pnns_groups_2"] = None

    df.head()

    df.describe(include="all")

    df.loc[df["product_name"].isna() | (df["product_name"].str.strip()=="")].shape

    df.drop(df.loc[(df["product_name"].isna()) | (df["product_name"].str.strip()=="")].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_parquet(f"{working_dir}/trimmed.parquet")
    df = pd.read_parquet(f"{working_dir}/trimmed.parquet")
    doublons = (
        df.assign(nb=1)
        .groupby(["product_name", "code"])
        .sum("nb")
        .reset_index()
        .query("nb>1")
        .loc[:, ["product_name", "code", "nb"]]
        .sort_values("nb", ascending=False)
        .reset_index(drop=True)
    )

    df["to_remove"] = False

    i = 1
    for ind in doublons.index:
        print(f"{i}/{doublons.shape[0]} ind={ind}")
        p = doublons.iloc[ind]["product_name"]
        b = doublons.iloc[ind]["code"]
        duplicated_indexes = df.loc[
            (df["product_name"] == p) & (df["code"] == b)
        ].index
        indexes_to_remove = cu.deduplicate(
            df=df,
            grouped_columns=["product_name", "code"],
            duplicated_indexes=duplicated_indexes,
            remove_doublons=False,
        )
        df.loc[indexes_to_remove, "to_remove"] = True
        i += 1
    df = df.loc[df["to_remove"] == False]
    df.drop(columns=["to_remove"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    doublons = (
        df.assign(nb=1)
        .groupby(["product_name", "brands_tags"])
        .sum("nb")
        .reset_index()
        .query("nb>1")
        .loc[:, ["product_name", "brands_tags", "nb"]]
        .sort_values("nb", ascending=False)
        .reset_index(drop=True)
    )
    df["to_remove"] = False

    i = 1
    for ind in doublons.index:
        print(f"{i}/{doublons.shape[0]} ind={ind}")
        p = doublons.iloc[ind]["product_name"]
        b = doublons.iloc[ind]["brands_tags"]
        duplicated_indexes = df.loc[
            (df["product_name"] == p) & (df["brands_tags"] == b)
        ].index
        indexes_to_remove = cu.deduplicate(
            df=df,
            grouped_columns=["product_name", "brands_tags"],
            duplicated_indexes=duplicated_indexes,
            remove_doublons=False,
        )
        df.loc[indexes_to_remove, "to_remove"] = True
        i += 1
    df = df.loc[df["to_remove"] == False]
    df.drop(columns=["to_remove"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_parquet(f"{working_dir}/deduplicated.parquet")

    df = pd.read_parquet(f"{working_dir}/deduplicated.parquet")
    df.describe(include="all")

    cols = ["fat_100g", "saturated-fat_100g", "carbohydrates_100g", "sugars_100g", "proteins_100g", "salt_100g", "sodium_100g"]
    cu.fix_outliers_from_ref_column_equality(
        df,
        ref_col="product_name",
        col_to_fix="fat_100g",
        lower_lim=0,
        upper_lim=100,
        verbose=False,
        show_progress=False,
    )

    cu.fix_outliers_from_ref_column_equality(
        df,
        ref_col="main_category",
        col_to_fix="fat_100g",
        lower_lim=0,
        upper_lim=100,
        verbose=False,
        show_progress=False,
    )

    p_n_terms = cu.extract_nb_terms_from_col(df, "product_name")
    imp_p_n_terms = {}
    for i,(k,v) in enumerate(p_n_terms.items()):
        if v>=100:
            imp_p_n_terms[k] = v

    for ind in df.loc[(df["fat_100g"]>100)].index:
        cu.fix_row_col_with_median_most_important_words(df, "product_name", ind, "fat_100g", imp_p_n_terms)

    df.drop(df.loc[(df["fat_100g"]<0) | (df["fat_100g"]>100)].index, inplace=True)
    df.reset_index(inplace=True, drop=True)

    cu.fix_outliers_from_ref_column_equality(
        df,
        ref_col="product_name",
        col_to_fix="saturated-fat_100g",
        lower_lim=0,
        upper_lim=100,
        verbose=False,
        show_progress=False,
    )
    cu.fix_outliers_from_ref_column_equality(
        df,
        ref_col="main_category",
        col_to_fix="saturated-fat_100g",
        lower_lim=0,
        upper_lim=100,
        verbose=False,
        show_progress=False,
    )
    cu.fix_outliers_from_ref_column_equality(
        df,
        ref_col="food_groups_tags",
        col_to_fix="saturated-fat_100g",
        lower_lim=0,
        upper_lim=100,
        verbose=False,
        show_progress=False,
    )

    for ind in df.loc[(df["saturated-fat_100g"]>100)].index:
        cu.fix_row_col_with_median_most_important_words(df, "product_name", ind, "saturated-fat_100g", imp_p_n_terms)

    fix_out_of_bounds_values(df, "carbohydrates_100g")

    df.drop(df.loc[(df["carbohydrates_100g"]<0) | (df["carbohydrates_100g"]>100)].index, inplace=True)
    df.reset_index(inplace=True, drop=True)

    fix_out_of_bounds_values(df, "sugars_100g")

    df.drop(df.loc[(df["sugars_100g"]<0) | (df["sugars_100g"]>100)].index, inplace=True)
    df.reset_index(inplace=True, drop=True)

    fix_out_of_bounds_values(df, "salt_100g")

    df.drop(df.loc[(df["salt_100g"]<0) | (df["salt_100g"]>100)].index, inplace=True)
    df.reset_index(inplace=True, drop=True)

    fix_out_of_bounds_values(df, "sodium_100g")

    fix_out_of_bounds_values(df, "proteins_100g")

    df.drop(df.loc[(df["proteins_100g"]<0) | (df["proteins_100g"]>100)].index, inplace=True)
    df.reset_index(inplace=True, drop=True)

    cols = ["fat_100g", "saturated-fat_100g", "carbohydrates_100g", "sugars_100g", "proteins_100g", "salt_100g", "sodium_100g"]

    for ind in df.loc[(df["fat_100g"]<df["saturated-fat_100g"]) & ~df["main_category"].isna()].index:
        for col_to_fix in ["fat_100g", "saturated-fat_100g"]:
            ref_value = df.loc[ind, "main_category"]
            cu.fix_var_with_median(df, "main_category", ref_value, ind, col_to_fix, verbose=False)

    cur_i = 0
    for ind in df.loc[df["fat_100g"]<df["saturated-fat_100g"]].index:
        for col_to_fix in ["fat_100g", "saturated-fat_100g"]:
            ref_value = df.loc[ind, "product_name"]
            cu.fix_var_with_median(df, "product_name", ref_value, ind, col_to_fix, verbose=False)
        cur_i += 1
        if cur_i % 20 == 0:
            print(f"{cur_i}/{tot}")

    for ind in df.loc[(df["fat_100g"]<df["saturated-fat_100g"]) & ~df["pnns_groups_2"].isna()].index:
        for col_to_fix in ["fat_100g", "saturated-fat_100g"]:
            ref_value = df.loc[ind, "pnns_groups_2"]
            cu.fix_var_with_median(df, "pnns_groups_2", ref_value, ind, col_to_fix, verbose=False)

    df.drop(df.loc[df["fat_100g"]<df["saturated-fat_100g"]].index, inplace=True)
    df.reset_index(inplace=True, drop=True)

    for ind in df.loc[(df["sugars_100g"]>df["carbohydrates_100g"]) & ~df["main_category"].isna()].index:
        for col_to_fix in ["sugars_100g", "carbohydrates_100g"]:
            ref_value = df.loc[ind, "main_category"]
            cu.fix_var_with_median(df, "main_category", ref_value, ind, col_to_fix, verbose=False)
    for ind in df.loc[(df["sugars_100g"]>df["carbohydrates_100g"]) & ~df["pnns_groups_2"].isna()].index:
        for col_to_fix in ["sugars_100g", "carbohydrates_100g"]:
            ref_value = df.loc[ind, "pnns_groups_2"]
            cu.fix_var_with_median(df, "pnns_groups_2", ref_value, ind, col_to_fix, verbose=False)

    tot = df.loc[df["sugars_100g"]>df["carbohydrates_100g"]].shape[0]
    cur_i = 0;
    for ind in df.loc[df["sugars_100g"]>df["carbohydrates_100g"]].index:
        for col_to_fix in ["sugars_100g", "carbohydrates_100g"]:
            ref_value = df.loc[ind, "product_name"]
            cu.fix_var_with_median(df, "product_name", ref_value, ind, col_to_fix, verbose=False)
        cur_i += 1
        if cur_i % 20 == 0:
            print(f"{cur_i}/{tot}")

    df.drop(df.loc[df["carbohydrates_100g"]<df["sugars_100g"]].index, inplace=True)
    df.reset_index(inplace=True, drop=True)

    df.to_parquet(f"{working_dir}/tmp.parquet")

    for ind in df.loc[((df["proteins_100g"]+df["fat_100g"]+df["carbohydrates_100g"]+df["salt_100g"])>100) & ~df["main_category"].isna()].index:
        for col_to_fix in ["proteins_100g", "fat_100g", "carbohydrates_100g", "salt_100g"]:
            ref_value = df.loc[ind, "main_category"]
            cu.fix_var_with_median(df, "main_category", ref_value, ind, col_to_fix, verbose=False)

    tot = df.loc[((df["proteins_100g"]+df["fat_100g"]+df["carbohydrates_100g"]+df["salt_100g"])>100)].shape[0]
    cur_i = 0;
    for ind in df.loc[((df["proteins_100g"]+df["fat_100g"]+df["carbohydrates_100g"]+df["salt_100g"])>100)].index:
        for col_to_fix in ["proteins_100g", "fat_100g", "carbohydrates_100g", "salt_100g"]:
            ref_value = df.loc[ind, "product_name"]
            cu.fix_var_with_median(df, "product_name", ref_value, ind, col_to_fix, verbose=False)
        cur_i += 1
        if cur_i % 20 == 0:
            print(f"{cur_i}/{tot}")
    df.to_parquet(f"{working_dir}/tmp.parquet")

    df.loc[~df["energy-kcal_100g"].isna() & ~df["energy_100g"].isna() & ~df["fat_100g"].isna() &
        ~df["carbohydrates_100g"].isna() & ~df["proteins_100g"].isna() &
        ((df["energy_100g"]/df["energy-kcal_100g"])<4.1) | ((df["energy_100g"]/df["energy-kcal_100g"])>4.2) , "energy-kcal_100g"] = 9* df["fat_100g"] + 4 * df["carbohydrates_100g"] + 4 * df["proteins_100g"]

    df.drop(columns="energy_100g", inplace=True)
    df.reset_index(inplace=True, drop=True)

    df.loc[df["energy-kcal_100g"]>1000, "energy-kcal_100g"] = 9 * df["fat_100g"] + 4 * df["carbohydrates_100g"] + 4 * df["proteins_100g"]

    df["masse"] = df["fat_100g"]+df["carbohydrates_100g"]+df["proteins_100g"]+df["salt_100g"]
    df.loc[(df["masse"]>100) & (df["masse"]<=105), "fat_100g"] = np.round(np.floor(100 * df["fat_100g"] / df["masse"]) / 10)
    df.loc[(df["masse"]>100) & (df["masse"]<=105), "carbohydrates_100g"] = np.round(np.floor(100 * df["carbohydrates_100g"] / df["masse"]) / 10)
    df.loc[(df["masse"]>100) & (df["masse"]<=105), "proteins_100g"] = np.round(np.floor(100 * df["proteins_100g"] / df["masse"]) / 10)
    df.loc[(df["masse"]>100) & (df["masse"]<=105), "salt_100g"] = np.round(np.floor(100 * df["salt_100g"] / df["masse"]) / 10)
    df.drop(columns=["masse"], inplace=True)

    df.drop(df.loc[((df["fat_100g"]+df["carbohydrates_100g"]+df["proteins_100g"]+df["salt_100g"])>100.1)].index, inplace=True)

    df.to_parquet(f"{working_dir}/after_aberrantes.parquet")
    df = pd.read_parquet(f"{working_dir}/after_aberrantes.parquet")


    cols=["energy-kcal_100g","fat_100g","saturated-fat_100g","carbohydrates_100g","sugars_100g","proteins_100g","salt_100g","sodium_100g"]

    for col in cols:
        pnns_groups_2 = df.loc[~df["pnns_groups_2"].isna() & df[col].isna()]["pnns_groups_2"].unique()
        for pnns2 in pnns_groups_2:
            tmp = df.loc[~df["pnns_groups_2"].isna() & ~df[col].isna() & (df["pnns_groups_2"]==pnns2), col]
            if tmp.shape[0]>0:
                median = np.median(tmp)
                df.loc[df[col].isna() & ~df["pnns_groups_2"].isna() & (df["pnns_groups_2"]==pnns2), col] = median


    cols=["energy-kcal_100g","fat_100g","saturated-fat_100g","carbohydrates_100g","sugars_100g","proteins_100g","salt_100g","sodium_100g"]
    nb_fixed = 0
    for col in cols:
        pnns_groups_1 = df.loc[~df["pnns_groups_1"].isna() & df[col].isna()]["pnns_groups_1"].unique()
        for pnns1 in pnns_groups_1:
            tmp = df.loc[~df["pnns_groups_1"].isna() & ~df[col].isna() & (df["pnns_groups_1"]==pnns1), col]
            if tmp.shape[0]>0:
                median = np.median(tmp)
                df.loc[df[col].isna() & ~df["pnns_groups_1"].isna() & (df["pnns_groups_1"]==pnns1), col] = median

    cols=["energy-kcal_100g","fat_100g","saturated-fat_100g","carbohydrates_100g","sugars_100g","proteins_100g","salt_100g","sodium_100g"]
    for col in cols:
        cur_ind = 1
        main_categories = df.loc[~df["main_category"].isna() & df[col].isna()]["main_category"].unique()
        for main_category in main_categories:
            tmp = df.loc[~df["main_category"].isna() & ~df[col].isna() & (df["main_category"]==main_category), col]
            if tmp.shape[0]>0:
                median = np.median(tmp)
                print(f"col={col}, main_category={main_category}, {cur_ind}/{len(main_categories)} median={median}")
                df.loc[df[col].isna() & ~df["main_category"].isna() & (df["main_category"]==main_category), col] = median
            cur_ind += 1
    df.to_parquet(f"{working_dir}/imputed_by_class.parquet")
    df = pd.read_parquet(f"{working_dir}/imputed_by_class.parquet")
    df.reset_index(inplace=True)

    df_impute = df.loc[~df["salt_100g"].isna() | ~df["sodium_100g"].isna(), ["salt_100g", "sodium_100g"]].copy(deep=True)
    imputer = IterativeImputer(estimator=LinearRegression(),missing_values=np.nan)
    df_imputed = pd.DataFrame(imputer.fit_transform(df_impute), columns=df_impute.columns, index=df_impute.index)


    df.loc[df_imputed.index,"salt_100g"] = df_imputed["salt_100g"]
    df.loc[df_imputed.index,"sodium_100g"] = df_imputed["sodium_100g"]

    df_impute = df.loc[~df["fat_100g"].isna() | ~df["energy-kcal_100g"].isna(), ["energy-kcal_100g", "fat_100g"]]
    df_impute = df_impute.copy(deep=True)
    imputer = IterativeImputer(estimator=LinearRegression(),missing_values=np.nan)
    df_imputed = pd.DataFrame(imputer.fit_transform(df_impute), columns=df_impute.columns, index=df_impute.index)
    df.loc[df_imputed.index,"energy-kcal_100g"] = df_imputed["energy-kcal_100g"]
    df.loc[df_imputed.index,"fat_100g"] = df_imputed["fat_100g"]

    df_impute = df.loc[~df["saturated-fat_100g"].isna() | ~df["fat_100g"].isna(), ["saturated-fat_100g", "fat_100g"]]
    df_impute = df_impute.copy(deep=True)
    imputer = IterativeImputer(missing_values=np.nan)
    df_imputed = pd.DataFrame(imputer.fit_transform(df_impute), columns=df_impute.columns, index=df_impute.index)
    df.loc[df_imputed.index,"saturated-fat_100g"] = df_imputed["saturated-fat_100g"]
    df.loc[df_imputed.index,"fat_100g"] = df_imputed["fat_100g"]

    df_impute = df.loc[:, ["sugars_100g", "carbohydrates_100g"]].copy(deep=True)
    imputer = IterativeImputer()
    df_imputed = pd.DataFrame(imputer.fit_transform(df_impute), columns=df_impute.columns)
    df["sugars_100g"] = df_imputed["sugars_100g"]
    df["carbohydrates_100g"] = df_imputed["carbohydrates_100g"]

    cols=["energy-kcal_100g","fat_100g","saturated-fat_100g","carbohydrates_100g","sugars_100g","proteins_100g","salt_100g","sodium_100g"]

    df.to_parquet(f"{working_dir}/imputed_iterative.parquet")
    df = pd.read_parquet(f"{working_dir}/imputed_iterative.parquet")

    indexes_to_fix = df.loc[(df["sugars_100g"] > df["carbohydrates_100g"])].index

    cur_nb = 0
    for index_to_fix in indexes_to_fix:
        fix_errors_values(df, index_to_fix, "sugars_100g")
        fix_errors_values(df, index_to_fix, "carbohydrates_100g")
        cur_nb += 1
        if cur_nb % 10:
            print(f"{cur_nb}/{len(indexes_to_fix)}")

    df.loc[(df["sugars_100g"] > df["carbohydrates_100g"])].shape

    df.loc[(df["saturated-fat_100g"] > df["fat_100g"])].shape

    indexes_to_fix = df.loc[(df["saturated-fat_100g"] > df["fat_100g"])].index

    cur_nb = 0
    for index_to_fix in indexes_to_fix:
        fix_errors_values(df, index_to_fix, "saturated-fat_100g")
        fix_errors_values(df, index_to_fix, "fat_100g")
        cur_nb += 1
        if cur_nb % 10:
            print(f"{cur_nb}/{len(indexes_to_fix)}")


    indexes_to_fix = df.loc[(df["fat_100g"] + df["carbohydrates_100g"] + df["proteins_100g"] + df["salt_100g"]) > 100].index

    cur_nb = 0
    for index_to_fix in indexes_to_fix:
        fix_errors_values(df, index_to_fix, "sugars_100g")
        fix_errors_values(df, index_to_fix, "carbohydrates_100g")
        fix_errors_values(df, index_to_fix, "fat_100g")
        fix_errors_values(df, index_to_fix, "salt_100g")
        cur_nb += 1
        if cur_nb % 10:
            print(f"{cur_nb}/{len(indexes_to_fix)}")


    df = df.loc[(df["sugars_100g"] <= df["carbohydrates_100g"])]
    df = df.loc[(df["saturated-fat_100g"] <= df["fat_100g"])]
    df = df.loc[(df["fat_100g"] + df["carbohydrates_100g"] + df["proteins_100g"] + df["salt_100g"]) <= 100]
    df.drop(columns="index", inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.to_parquet(f"{working_dir}/imputed_errors_fixed3.parquet")
    df = pd.read_parquet(f"{working_dir}/imputed_errors_fixed3.parquet")
    df.reset_index(inplace=True, drop=True)
    df.drop(columns=["completeness"], inplace=True)
    df.loc[
        ~df["nutriscore_grade"].isna() &
        ~df["fat_100g"].isna() &
        ~df["energy-kcal_100g"].isna() &
        ~df["saturated-fat_100g"].isna() &
        ~df["carbohydrates_100g"].isna() &
        ~df["sugars_100g"].isna() &
        ~df["proteins_100g"].isna() &
        ~df["salt_100g"].isna() &
        ~df["sodium_100g"].isna()
    ].shape
    df.loc[
        df["nutriscore_grade"].isna() &
        ~df["fat_100g"].isna() &
        ~df["energy-kcal_100g"].isna() &
        ~df["saturated-fat_100g"].isna() &
        ~df["carbohydrates_100g"].isna() &
        ~df["sugars_100g"].isna() &
        ~df["proteins_100g"].isna() &
        ~df["salt_100g"].isna() &
        ~df["sodium_100g"].isna()
    ].shape

    knn = neighbors.KNeighborsClassifier(1)
    df_knn = df.loc[
        ~df["nutriscore_grade"].isna() &
        ~df["fat_100g"].isna() &
        ~df["energy-kcal_100g"].isna() &
        ~df["saturated-fat_100g"].isna() &
        ~df["carbohydrates_100g"].isna() &
        ~df["sugars_100g"].isna() &
        ~df["proteins_100g"].isna() &
        ~df["salt_100g"].isna() &
        ~df["sodium_100g"].isna()
    ]
    data = df_knn.loc[:, ['energy-kcal_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'proteins_100g', 'salt_100g', 'sodium_100g']]
    target = df_knn.loc[:, "nutriscore_grade"]
    knn.fit(data, target)

    predicted = knn.predict(
        df.loc[
            df["nutriscore_grade"].isna() &
            ~df["fat_100g"].isna() &
            ~df["energy-kcal_100g"].isna() &
            ~df["saturated-fat_100g"].isna() &
            ~df["carbohydrates_100g"].isna() &
            ~df["sugars_100g"].isna() &
            ~df["proteins_100g"].isna() &
            ~df["salt_100g"].isna() &
            ~df["sodium_100g"].isna(),
            ['energy-kcal_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'proteins_100g', 'salt_100g', 'sodium_100g']
        ]
    )

    df.loc[
        df["nutriscore_grade"].isna() &
        ~df["fat_100g"].isna() &
        ~df["energy-kcal_100g"].isna() &
        ~df["saturated-fat_100g"].isna() &
        ~df["carbohydrates_100g"].isna() &
        ~df["sugars_100g"].isna() &
        ~df["proteins_100g"].isna() &
        ~df["salt_100g"].isna() &
        ~df["sodium_100g"].isna(),
        ['nutriscore_grade']
    ] = predicted

    knn = neighbors.KNeighborsClassifier(1)
    df_knn = df.loc[
        ~df["pnns_groups_2"].isna() &
        ~df["fat_100g"].isna() &
        ~df["energy-kcal_100g"].isna() &
        ~df["saturated-fat_100g"].isna() &
        ~df["carbohydrates_100g"].isna() &
        ~df["sugars_100g"].isna() &
        ~df["proteins_100g"].isna() &
        ~df["salt_100g"].isna() &
        ~df["sodium_100g"].isna()
    ]
    data = df_knn.loc[:, ['energy-kcal_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'proteins_100g', 'salt_100g', 'sodium_100g']]
    target = df_knn.loc[:, "pnns_groups_2"]
    knn.fit(data, target)

    predicted = knn.predict(
        df.loc[
            df["pnns_groups_2"].isna() &
            ~df["fat_100g"].isna() &
            ~df["energy-kcal_100g"].isna() &
            ~df["saturated-fat_100g"].isna() &
            ~df["carbohydrates_100g"].isna() &
            ~df["sugars_100g"].isna() &
            ~df["proteins_100g"].isna() &
            ~df["salt_100g"].isna() &
            ~df["sodium_100g"].isna(),
            ['energy-kcal_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'proteins_100g', 'salt_100g', 'sodium_100g']
        ]
    )

    df.loc[
        df["pnns_groups_2"].isna() &
        ~df["fat_100g"].isna() &
        ~df["energy-kcal_100g"].isna() &
        ~df["saturated-fat_100g"].isna() &
        ~df["carbohydrates_100g"].isna() &
        ~df["sugars_100g"].isna() &
        ~df["proteins_100g"].isna() &
        ~df["salt_100g"].isna() &
        ~df["sodium_100g"].isna(),
        ['pnns_groups_2']
    ] = predicted

    df.drop(columns=["main_category"], inplace=True)

    knn = neighbors.KNeighborsClassifier(1)
    df_knn = df.loc[
        ~df["food_groups_tags"].isna() &
        ~df["fat_100g"].isna() &
        ~df["energy-kcal_100g"].isna() &
        ~df["saturated-fat_100g"].isna() &
        ~df["carbohydrates_100g"].isna() &
        ~df["sugars_100g"].isna() &
        ~df["proteins_100g"].isna() &
        ~df["salt_100g"].isna() &
        ~df["sodium_100g"].isna()
    ]
    data = df_knn.loc[:, ['energy-kcal_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'proteins_100g', 'salt_100g', 'sodium_100g']]
    target = df_knn.loc[:, "food_groups_tags"]
    knn.fit(data, target)

    predicted = knn.predict(
        df.loc[
            df["food_groups_tags"].isna() &
            ~df["fat_100g"].isna() &
            ~df["energy-kcal_100g"].isna() &
            ~df["saturated-fat_100g"].isna() &
            ~df["carbohydrates_100g"].isna() &
            ~df["sugars_100g"].isna() &
            ~df["proteins_100g"].isna() &
            ~df["salt_100g"].isna() &
            ~df["sodium_100g"].isna(),
            ['energy-kcal_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'proteins_100g', 'salt_100g', 'sodium_100g']
        ]
    )

    df.loc[
        df["food_groups_tags"].isna() &
        ~df["fat_100g"].isna() &
        ~df["energy-kcal_100g"].isna() &
        ~df["saturated-fat_100g"].isna() &
        ~df["carbohydrates_100g"].isna() &
        ~df["sugars_100g"].isna() &
        ~df["proteins_100g"].isna() &
        ~df["salt_100g"].isna() &
        ~df["sodium_100g"].isna(),
        ['food_groups_tags']
    ] = predicted


    pnns_groups = {}
    for pnns1 in df["pnns_groups_1"].unique():
        if pd.isna(pnns1):
            continue
        if not pnns1 in pnns_groups:
            pnns_groups[pnns1] = []
        for pnns2 in df.loc[df["pnns_groups_1"]==pnns1]["pnns_groups_2"].unique():
            pnns_groups[pnns1].append(pnns2)

    df["pnns_groups_1"] = df.apply(lambda x: fill_pnns_1(x), axis=1)
    df.to_parquet(f"{working_dir}/after_knn_clean.parquet")
    df = pd.read_parquet(f"{working_dir}/after_knn_clean.parquet")
        
    df.drop(columns=["brands_tags"], inplace=True) 

    df.query("&".join(map(lambda x: f""" `{x}`.notnull() """, list(df.columns)))).reset_index(drop=True).to_parquet(output_filename)

def print_usage(exit_code=0):
    print(f"""Usage: python3 {sys.argv[0]} --input|-i input_file_name --output|-o output_file_name""")
    exit(exit_code)
    
def main():
    args = sys.argv[1:]
    input_filename = None
    output_filename = None
    for i in range(0, len(args)):
        if len(args)>i+1 and (args[i]=="--input" or args[i]=="-i"):
            input_filename = args[i+1]
        if len(args)>i+1 and (args[i]=="--output" or args[i]=="-o"):
            output_filename = args[i+1]
        if args[i]=="-h" or args[i]=="--help":
            print_usage()
            
    if input_filename is None or output_filename is None:
        print_usage(0)
        
    if os.path.exists(input_filename)==False:
        print(f"Input file {input_filename} does not exist")
        exit(1)
    
    clean_data(input_filename, output_filename)
    
    
if __name__ == "__main__":
    main()
