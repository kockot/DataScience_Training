import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from lib.customTransformers import (
    catFeaturesNone2NaNTransformer,
    featureCopyTransformer,
    featureDropTransformer,
    featureDateTimeTransformer,
    featureSplitterTransformer,
    weekdayTransformer,
    dateAgeTransformer,
    renameColumnTransformer,
    numFeaturesImputerTransformer,
    catFeaturesImputerTransformer,
    numFeaturesScalerTransformer,
    catFeaturesEncoderTransformer,
    featureDropAllButTransformer,
    mostCommonWeekdayTransformer,
    featureMultiplierTransformer,
    mostPurchasedTransformer,
    featureSubstractTransformer,
    DataFrameOneHotEncoder,
    minValueTransformer,
    debugTransformer
)


def cat_mapper(c, prefix):
    cat_mapping = {
        "bed_bath_table": "home_office",
        "health_beauty": "health_beauty",
        "stationery": "home_office",
        "telephony": "electronics",
        "garden_tools": "home_office",
        "sports_leisure": "clothes",
        "fashion_bags_accessories": "clothes",
        "luggage_accessories": "clothes",
        "computers_accessories": "electronics",
        "fashion_underwear_beach": "clothes",
        "home_appliances": "electronics",
        "musical_instruments": "culture_leisure",
        "toys": "culture_leisure",
        "home_confort": "home_office",
        "home_confort_2": "home_office",
        "housewares": "home_office",
        "small_appliances": "electronics",
        "watches_gifts": "cool_stuff",
        "electronics": "electronics",
        "furniture_living_room": "home_office",
        "office_furniture": "home_office",
        "auto": "other",
        "furniture_decor": "home_office",
        "perfumery": "health_beauty",
        "construction_tools_construction": "construction",
        "drinks": "food_drink",
        "books_general_interest": "culture_leisure",
        "consoles_games": "electronics",
        "cool_stuff": "cool_stuff",
        "christmas_supplies": "cool_stuff",
        "baby": "health_beauty",
        "pet_shop": "other",
        "home_construction": "construction",
        "home_appliances_2": "electronics",
        "fixed_telephony": "electronics",
        "flowers": "home_office",
        "books_imported": "culture_leisure",
        "construction_tools_safety": "construction",
        "diapers_and_hygiene": "health_beauty",
        "fashion_shoes": "clothes",
        "kitchen_dining_laundry_garden_furniture": "home_office",
        "art": "culture_leisure",
        "food_drink": "food_drink",
        "food": "food_drink",
        "books_technical": "culture_leisure",
        "industry_commerce_and_business": "other",
        "audio": "electronics",
        "construction_tools_lights": "construction",
        "signaling_and_security": "construction",
        "market_place": "other",
        "fashion_male_clothing": "clothes",
        "cine_photo": "culture_leisure",
        "costruction_tools_garden": "construction",
        "agro_industry_and_commerce": "other",
        "furniture_bedroom": "home_office",
        "dvds_blu_ray": "culture_leisure",
        "costruction_tools_tools": "construction",
        "fashion_sport": "clothes",
        "computers": "electronics",
        "furniture_mattress_and_upholstery": "home_office",
        "home_comfort": "home_office",
        "home_comfort_2": "home_office",
        "air_conditioning": "construction",
        "tablets_printing_image": "electronics",
        "cds_dvds_musicals": "culture_leisure",
        "fashio_female_clothing": "clothes",
        "party_supplies": "cool_stuff",
        "small_appliances_home_oven_and_coffee": "home_office",
        "music": "culture_leisure",
        "arts_and_craftmanship": "construction",
        "security_and_services": "other",
        "fashion_childrens_clothes": "clothes",
        "la_cuisine": "home_office",
    }
    r = "NA"
    if c in cat_mapping.keys():
        r = cat_mapping[c]
    r = f"{prefix}{r}"
    return r


def payment_type_mapper(c, prefix):
    payment_mapping = {
        "credit_card": "credit_card",
        "boleto": "cash",
        "voucher": "voucher",
        "debit_card": "debit_card",
        "not_defined": "not_defined",
    }
    r = "not_defined"
    if c in payment_mapping.keys():
        r = payment_mapping[c]
    r = f"{prefix}{r}"
    return r


def clean(df, use_features=[],use_standard_scaler=False, use_min_max_scaler=True, pipe=None):
    #df.dropna(inplace=True)
    
    if pipe is None:
        keep_exact_columns = ["r", "f", "m"]
        keep_columns_starting_with = []

        get_last = True if ("last_timestamp" in use_features or "last_amount" in use_features) else False
        if "categories" in use_features:
            keep_columns_starting_with.append("all_cats_")
            if get_last:
                keep_columns_starting_with.append("last_cat_")
        if "payment_types" in use_features:
            keep_columns_starting_with.append("all_payments_")
            if get_last:
                keep_columns_starting_with.append("last_payment_")
        if "weekday" in use_features:
            keep_exact_columns.append("most_purchased_weekday")
            if get_last:
                keep_exact_columns.append("last_timestamp_weekday")
        if "review" in use_features:
            keep_exact_columns.append("all_purchases_review_score")
            if get_last:
                keep_exact_columns.append("last_review_score")
        if "most_purchased_category" in use_features:
            keep_exact_columns.append("all_most_purchased_category")
            if get_last:
                keep_exact_columns.append("last_most_purchased_category")
        if "most_payment_type" in use_features:
            keep_exact_columns.append("all_most_payment_type")
            if get_last:
                keep_exact_columns.append("last_most_payment_type")
        if "delivery" in use_features:
            keep_exact_columns.append("all_delivery_delay")
            if get_last:
                keep_exact_columns.append("last_delivery_delay")
        if "freight_value" in use_features:
            keep_exact_columns.append("all_freight_value")
            if get_last:
                keep_exact_columns.append("last_freight_value")

        if "photos_quantity" in use_features:
            keep_exact_columns.append("all_photos_quantity")
            if get_last:
                keep_exact_columns.append("last_photos_quantity")

        if get_last:
            keep_exact_columns.append("last_timestamp")
            keep_exact_columns.append("last_amount")


        if "customer_state" in use_features:
            keep_exact_columns.append("customer_state")
        if "customer_location" in use_features:
            keep_exact_columns.append("customer_latitude")
            keep_exact_columns.append("customer_longitude")
        if "customer_unique_id" in use_features:
            keep_exact_columns.append("customer_unique_id")

        if "first_purchase_timestamp" in use_features:
            keep_exact_columns.append("first_purchase_timestamp")

        min_last_timestamp = np.datetime64(df["last_timestamp"].min())
        max_last_timestamp = np.datetime64(df["last_timestamp"].max())
        duration = ((max_last_timestamp - min_last_timestamp)/np.timedelta64(1, 'D'))

        numerical_imputer =   SimpleImputer(strategy="median")
        categorical_imputer = SimpleImputer(strategy="most_frequent")

        steps = [
            ("none2na", catFeaturesNone2NaNTransformer(df.select_dtypes(["object", "float64", "int64"]))),
            ("num_imputer", numFeaturesImputerTransformer(imputer=numerical_imputer)),
            ("cat_imputer", catFeaturesImputerTransformer(imputer=categorical_imputer)),
            (
                "drop_unused_ids",
                featureDropTransformer(["last_order_id"]),
            ),
            ("dt_last_timestamp", featureDateTimeTransformer(["last_timestamp"])),
            ("copy_last_timestamp", featureCopyTransformer("last_timestamp", "last_timestamp2")),
            ("create_first_purchase_timestamp", minValueTransformer("all_purchases_timestamps", "first_purchase_timestamp")),
            ("dt_first_timestamp", featureDateTimeTransformer(["first_purchase_timestamp"])),
            (
                "create_last_purchase_cat_cols",
                featureSplitterTransformer(
                    "last_categories",
                    split_char_1=",",
                    split_char_2=":",
                    prefix="last_cat_",
                    splitter_fn=cat_mapper,
                ),
            ),
            (
                "create_all_purchases_cat_cols",
                featureSplitterTransformer(
                    "all_purchases_categories",
                    split_char_1=",",
                    split_char_2=":",
                    prefix="all_cats_",
                    splitter_fn=cat_mapper,
                ),
            ),
            (
                "create_last_payment_type_cols",
                featureSplitterTransformer(
                    "last_payments",
                    split_char_1=",",
                    split_char_2=":",
                    prefix="last_payment_",
                    splitter_fn=payment_type_mapper,
                ),
            ),
            (
                "create_all_payment_types_cols",
                featureSplitterTransformer(
                    "all_purchases_payments",
                    split_char_1=",",
                    split_char_2=":",
                    prefix="all_payments_",
                    splitter_fn=payment_type_mapper,
                ),
            ),
            (
                "last_purchase_weekday",
                weekdayTransformer("last_timestamp", "last_timestamp_weekday"),
            ),
            (
                "most_purchased_weekday",
                mostCommonWeekdayTransformer(
                    "all_purchases_timestamps", "most_purchased_weekday"
                ),
            ),
            (
                "all_most_purchased_category",
                mostPurchasedTransformer(
                    "all_cats_", "all_most_purchased_category"
                ),
            ),
            (
                "last_most_purchased_category",
                mostPurchasedTransformer(
                    "last_cat_", "last_most_purchased_category"
                ),
            ),
            (
                "all_most_payments_types",
                mostPurchasedTransformer(
                    "all_payments_", "all_most_payment_type"
                ),
            ),
            (
                "last_most_paymens_type",
                mostPurchasedTransformer(
                    "last_payment_", "last_most_payment_type"
                ),
            ),
            (
                "last_delivery_delay_estimated_real",
                featureSubstractTransformer(
                    "last_effective_delivery_delay", "last_expected_delivery_delay", "last_delivery_delay"
                )
            ),
            (
                "delivery_delay_estimated_real",
                featureSubstractTransformer(
                    "all_effective_delivery_delay", "all_expected_delivery_delay", "all_delivery_delay"
                )
            ),
            #(
            #    "drop_purchase_categories",
            #    featureDropTransformer(
            #        [
            #            "last_categories",
            #            "all_purchases_categories",
            #            "last_payments",
            #            "all_purchases_payments",
            #        ]
            #    ),
            #),
            ("create_r_column", dateAgeTransformer("last_timestamp2", "r", max_last_timestamp)),
            ("create_f_column", renameColumnTransformer("nb_all_purchases", "f")),
            ("create_f_frequency", featureMultiplierTransformer("f", "f", 365/duration)),
            ("create_m_column", renameColumnTransformer("all_purchases_amount", "m")),
            (
                "keep_columns",
                featureDropAllButTransformer(
                    keep_exact_columns=keep_exact_columns,
                    keep_columns_starting_with=keep_columns_starting_with,
                ),
            ),
        ]


        pipe = Pipeline(steps=steps)
        data0 = pipe.fit_transform(df)
    else:
        data0 = pipe.transform(df)
        
    scaler = None
    if use_standard_scaler or use_min_max_scaler:
        num_cols = []
        cat_cols = []
        for c in data0.columns:
            if data0.dtypes[c]=="float64" or data0.dtypes[c]=="int64":
                num_cols.append(c)
            else:
                cat_cols.append(c)
        
        #data0 = pd.get_dummies(data0)
        if use_standard_scaler:
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        data0_num = pd.DataFrame(scaler.fit_transform(data0.loc[:, num_cols]), columns = num_cols)
        data0_cat = data0.loc[:, cat_cols]
        data0 = pd.concat([data0_num, data0_cat],axis=1, join='inner')
    else:
        scaler=None
        
    
    data0.reset_index(drop=True, inplace=True)


    r = {
        "data": data0, 
        "scaler": scaler,
        "pipe": pipe,
    }
    return r