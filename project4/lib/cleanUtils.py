import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import neighbors

import plotly.graph_objects as go
import plotly.express as px
import itertools


def fix_var_with_median(
    df, ref_col_name, ref_value, index_to_fix, col_to_fix, verbose=False
):
    """
    This function will fix a specific variable *col_to_fix* value for a specific index *index_to_fit* in a DataFrame *df* with the median value of the variable *col_to_fix*
    for all the observations (except the one at index *index_to_fit*) having their variables *ref_col_name* set to the value *ref_value*.

        Parameters:
            df (pandas.DataFrame): DataFrame to fix
            ref_col_name (string): variable name to search for to get the median
            ref_value (string): variable value to search for to get the median
            index_to_fix (int): index in df to fix the value
            col_to_fix (string): variable name of the cell to fix
            verbose (boolean): optionnal boolean to show output verbose. Defualts to False

        Returns:
            Float|None: median value used to replace the cell value
    """
    others = df.loc[
        (df.index != index_to_fix)
        & ~df[col_to_fix].isna()
        & (df[ref_col_name].str.lower() == str(ref_value).lower())
    ]
    if others.shape[0] > 0:
        m = others[col_to_fix].median()
        if verbose:
            print(
                f"fix_var_with_median will change dataframe at index {index_to_fix}, column {col_to_fix} to value : {m}"
            )
        df.loc[index_to_fix, col_to_fix] = m
        return m

    if verbose:
        print(
            f"""WARNING: fix_var_with_median could not find other values where {ref_col_name}="{ref_value}" to compute median"""
        )
    return None


def fix_var_with_median_ref_indexes(
    df, ref_indexes, index_to_fix, col_to_fix, verbose=False
):
    """
    This function will fix a specific variable *col_to_fix* value for a specific index *index_to_fix* in a DataFrame *df* with the median value of the variable *col_to_fix*
    for all the observations (except the one at index *index_to_fit*) having their variables *ref_col_name* set to the value *ref_value*.

        Parameters:
            df (pandas.DataFrame): DataFrame to fix
            ref_indexes (list()): list of rows indexes to get the median from
            index_to_fix (int): index in df to fix the value
            col_to_fix (string): variable name of the cell to fix
            verbose (boolean): optionnal boolean to show output verbose. Defualts to False

        Returns:
            Float|None: median value used to replace the cell value
    """
    other_indexes = list(set(ref_indexes) - set([index_to_fix]))
    others = df.iloc[other_indexes]
    others = others.loc[~others[col_to_fix].isna()]
    if others.shape[0] > 0:
        m = others[col_to_fix].median()
        if verbose:
            print(
                f"fix_var_with_median will change dataframe at index {index_to_fix}, column {col_to_fix} to value : {m}"
            )
        df.loc[index_to_fix, col_to_fix] = m
        return m

    if verbose:
        print(
            f"""WARNING: fix_var_with_median_ref_indexes could not find other values where index in {ref_indexes}" to compute median"""
        )
    return None


def fix_var_with_median_ref_indexes_with_ref_df(
    df, df_ref, ref_indexes, index_to_fix, col_to_fix, verbose=False
):
    """
    This function will fix a specific variable *col_to_fix* value for a specific index *index_to_fix* in a DataFrame *df* with the median value of the variable *col_to_fix*
    for all the observations (except the one at index *index_to_fit*) having their variables *ref_col_name* set to the value *ref_value*.

        Parameters:
            df (pandas.DataFrame): DataFrame to fix
            df_ref (pandas.DataFrame): DataFrame to search median in
            ref_indexes (list()): list of rows indexes to get the median from
            index_to_fix (int): index in df to fix the value
            col_to_fix (string): variable name of the cell to fix
            verbose (boolean): optionnal boolean to show output verbose. Defualts to False

        Returns:
            Float|None: median value used to replace the cell value
    """
    #other_indexes = list(set(ref_indexes) - set([index_to_fix]))
    others = df_ref.loc[ref_indexes]
    others = others.loc[~others[col_to_fix].isna()]
    if others.shape[0] > 0:
        m = others[col_to_fix].median()
        if verbose:
            print(
                f"fix_var_with_median will change dataframe at index {index_to_fix}, column {col_to_fix} to value : {m}"
            )
        df.loc[index_to_fix, col_to_fix] = m
        return m

    if verbose:
        print(
            f"""WARNING: fix_var_with_median_ref_indexes could not find other values where index in {ref_indexes}" to compute median"""
        )
    return None


def _extract_from_text(s, sep=","):
    l = s.split(sep)
    for i in range(0, len(l)):
        l[i] = l[i].strip()
    return l


def get_nb_tags_from_string(s, sep=","):
    """
    Returns the number of different tags in a string s separated by the string sep

        Parameters:
            s (string): input string
            sep (string): separator string

        Returns:
            int: number of different tags
    """
    if pd.isna(s):
        return np.NaN
    s = str(s)
    tags = dict()
    for e in s.split(sep):
        e = e.strip()
        if not e in tags:
            tags[e] = True
    return len(tags)


def extract_nb_terms_from_col(data, col, sep=[" ", ",", "-"], min_length=4):
    """
    Goes throught all the values for a column col in a dataframe data and counts number of words separated by the strings in the sep array.
    Words whose length are strictly lower than min_length are not taken

        Parameters:
            data (pandas.Dataframe)
            col (string): column name to consider
            sep (array|list): list of serarator strings
            min_length (int): minimum word length to take

        Returns:
            (dict): dictionnary where key is the term string (lower case) and value is the term occurence number
    """
    tags = dict()
    for e_l in data.loc[~data[col].isna()][col].str.lower():

        for i in range(1, len(sep)):
            e_l = e_l.replace(sep[i], sep[0])
        local_tags = dict()  # just to avoid local doublons inside the same cell
        for e in e_l.split(sep[0]):
            e = e.strip().lower()
            if len(e) < min_length:
                continue
            if not e in local_tags:
                local_tags[e] = True
                if not e in tags:
                    tags[e] = 1
                else:
                    tags[e] += 1
    return dict(sorted(tags.items(), key=lambda x: x[1], reverse=True))


def get_most_used_terms_from_string(s, most_used_terms, sep=[" ", ",", "-"]):
    """
    Returns the list of words in string s that belong to the dictionnary of terms most_used_terms (as returned by extract_nb_terms_from_col), returns the words.
    Words are separated by strings defined in the sep array

        Parameters:
            s (string): original string
            most_used_terms (dict) dictionnary where key is the term string (lower case), value is not important here
            sep (array|list): list of serarator strings

        Returns:
            (array) array of words

    """
    s = s.lower()
    for i in range(1, len(sep)):
        s = s.replace(sep[i], sep[0])
    l = s.split(sep[0])
    ret = []
    for i in range(0, len(l)):
        l[i] = l[i].lower().strip()
        if l[i] in most_used_terms:
            ret.append(l[i])
    return ret


def fix_row_col_with_median_most_important_words(
    df: pd.DataFrame,
    ref_col: str,
    index_to_fix: int,
    col_to_fix: str,
    most_used_globally_terms: dict = None,
    verbose: bool = False,
):
    """
    This function will fix a specific variable *col_to_fix* value for a specific index *index_to_fix* in a DataFrame *df* with the median value of the variable *col_to_fix*
    for all the observations (except the one at index *index_to_fit*) having their variables *ref_col_name* most used words belonging to the *most_used_globally_terms* dictionnary keys.
    That dictionnary can be retrieved by the function extract_nb_terms_from_col and trimmed to a subset.

        Parameters:
            df (pandas.DataFrame): DataFrame to fix
            ref_col (str): column where to search for terms
            index_to_fix (int): index in df to fix the value
            col_to_fix (string): variable name of the cell to fix
            most_used_globally_terms (dict): dictionnary where leys are the words we define as the most used to search for
            verbose (boolean): optionnal boolean to show output verbose. Defualts to False

        Returns:
            Float|None: median value used to replace the cell value
    """
    if most_used_globally_terms is None:
        if ref_col is None:
            raise Exception(
                "When most_used_globally_terms is not set or is None, you have to provide the referenced column with ref_col parameter"
            )
        elif ref_col not in df.columns:
            raise Exception(f"""Column {ref_col} does not exists in dataframe""")
        global_terms = extract_nb_terms_from_col(df, "product_name")
        most_used_globally_terms = {}
        for i, (k, v) in enumerate(global_terms.items()):
            if v >= 100:
                most_used_globally_terms[k] = v

    ref_val = df.iloc[index_to_fix][ref_col]
    if ref_val is None:
        return None
    ref_val_words = get_most_used_terms_from_string(ref_val, most_used_globally_terms)
    q = ""
    for i in range(0, len(ref_val_words)):
        if i > 0:
            q += " & "
        q += f""" {ref_col}.str.lower().str.contains("{ref_val_words[i]}") """
    if q.strip() != "":
        q = f"{ref_col}.notnull() & " + q
        ref_indexes = df.query(q).index
        m = fix_var_with_median_ref_indexes(
            df, ref_indexes, index_to_fix, col_to_fix, verbose
        )
        if verbose:
            if m is None:
                print("no change")
            else:
                print(f"changed to {m}")
        return m
    return None


def fix_outliers_from_ref_column_equality(
    df,
    ref_col=None,
    col_to_fix=None,
    lower_lim=None,
    upper_lim=None,
    verbose=False,
    show_progress=False,
):
    """
    Fixes values in fataframe *df* for column *col_to_fix* for values below *lower_lim*
    or values greater than *upper_lim* by calling function fix_var_with_median with *ref_col*
        Parameters:
            df (pandas.DataFrame): DataFrame to fix
            ref_col (str): column where to search for terms
            col_to_fix (string): variable name of the cell to fix
            lower_lim (Float): if not None, tries to fix for values < lower_limit
            upper_lim (Float): if not None, tries to fix for values > upper_limit
            verbose (boolean): optionnal boolean to show output verbose. Defualts to False
            show_progress (boolean)
        Returns:
            Float|None: median value used to replace the cell value

    """
    if lower_lim is not None:
        nb_tot = 0
        nb_cur = 0
        if show_progress:
            nb_tot = df.loc[~df[ref_col].isna() & (df[col_to_fix] < lower_lim)].shape[0]
        for ind in df.loc[~df[ref_col].isna() & (df[col_to_fix] < lower_lim)].index:
            if show_progress:
                nb_cur += 1
                print(f"fixing lower lim {nb_cur}/{nb_tot}")
            ref_value = df.loc[ind, ref_col]
            fix_var_with_median(
                df, ref_col, ref_value, ind, col_to_fix, verbose=verbose
            )

    if upper_lim is not None:
        nb_tot = 0
        nb_cur = 0
        if show_progress:
            nb_tot = df.loc[~df[ref_col].isna() & (df[col_to_fix] > upper_lim)].shape[0]
        for ind in df.loc[~df[ref_col].isna() & (df[col_to_fix] > upper_lim)].index:
            if show_progress:
                nb_cur += 1
                print(f"fixing upper lim {nb_cur}/{nb_tot}")
            ref_value = df.loc[ind, ref_col]
            fix_var_with_median(
                df, ref_col, ref_value, ind, col_to_fix, verbose=verbose
            )


def deduplicate(
    df, grouped_columns=[], duplicated_indexes=[], remove_doublons=True, cols=[]
):
    """
    Given a set of grouped observation indexes duplicated_indexes in a dataframe df,
    this function tries to fill NA values in the observation at df.loc[duplicated_indexes[0]]
    from df.loc[duplicated_indexes[1:]]

    For string cells, if the length is greater in df.loc[duplicated_indexes[1:]], then the contents is copied to
    df.loc[duplicated_indexes[0]] (greater string length is assumed to mean more information)

    If remove_doublons is set to True, the observations at indexes duplicated_indexes[1:] are removed

    Parameters:
        df (DataFrame): Initial dataframe
        grouped_columns (list(string)): variable names used to group the observations
        duplicated_indexes (list(int)): dataframe indexes of grouped observations
        remove_doublons (bool): specifies whether or not to delete the observations at duplicated_indexes[1:]
        cols (list(string)): list of dataframe columns - optional, only used to speed up the function
    Returns:
        (array): list of indexes to remove or removed (if remove_doublons True)
    """
    to_remove_indexes = []
    if len(cols) == 0:
        cols = df.columns

    for y in range(1, len(duplicated_indexes)):
        for x in range(0, len(cols)):
            if cols[x] in grouped_columns:
                continue

            if (
                pd.isna(df.iat[duplicated_indexes[0], x]) == True
                and pd.isna(df.iat[duplicated_indexes[y], x]) == False
            ):
                df.iat[duplicated_indexes[0], x] = df.iat[duplicated_indexes[y], x]
            elif (
                False
                and pd.isna(df.iat[duplicated_indexes[0], x]) == False
                and pd.isna(df.iat[duplicated_indexes[y], x]) == False
                and df.dtypes[x] == "object"
            ):
                # On compare les longueurs pour prendre la cellule qui a le plus grand contenu
                try:
                    # pandas considère les timestamps et int comme des object de la même manière que les string
                    if len(df.iat[duplicated_indexes[y], x]) > len(
                        df.iat[duplicated_indexes[0], x]
                    ):
                        df.iat[duplicated_indexes[0], x] = df.iat[
                            duplicated_indexes[y], x
                        ]
                except:
                    pass
        to_remove_indexes.append(duplicated_indexes[y])
        if remove_doublons == True:
            df.drop(duplicated_indexes[y], inplace=True)

    return to_remove_indexes


def evaluate_KNNImputer(
    df, cols=None, cols_to_impute=None, nb_samples=5000, ratio_nan=0.25, k_to_test=range(1, 30)
):
    """ 
    Plots KNNImputer evaluation for a given dataframe df with columns cols. The plot will work on a random sample of nb_samples fully filled rows,
    and insert a ratio of null values in each column of ratio_nan. This works among different values of the k parameter given in k_to_test.
    The plot show the mean error and the max for each k and column.
    Parameters:
        df(pandas.DataFrame): dataframe to run evaluation on
        cols(list|array): columns of the parameter to take into account for the neighbourhood. By default, it will grab all columns with dtype not equal to 'O'
        cols_to_impute(list|array): columns of the parameter to take into account for the imputation. By default, it will grab all columns with dtype not equal to 'O'
        nb_samples(int): sample length of initial dataframe to work with
        ratio_nan(float): ratio of missing values to insert in the testing sample for each column (of cols_to_impute)
        k_to_test(list|array|iterator): set of k values to test against
    """
    if cols == None:
        cols = []
        for c in df.columns:
            if c.columns.dtype != "O":
                cols.append(c)

    if cols_to_impute == None:
        cols = []
        for c in df.columns:
            if c.columns.dtype != "O":
                cols.append(c)

    nb_samples_nan = round(nb_samples * ratio_nan)

    df_trimmed = (
        df.query("&".join(map(lambda x: f""" `{x}`.notnull() """, list(cols))))
        .loc[:, cols]
        .copy()
        
    )

    df_orig = df_trimmed.sample(n=nb_samples).reset_index()
    df_test = df_orig.copy()

    for c in cols_to_impute:
        df_test[c] = df_test[c].sample(n=nb_samples - nb_samples_nan)

    df_hist = pd.DataFrame(columns=["n", "column", "diff", "diff_relative", "orig", "imputed"])

    for n in k_to_test:
        imputer = KNNImputer(n_neighbors=n)
        scaler = MinMaxScaler()
        scaled_input = scaler.fit_transform(df_test)
        scaled_output = imputer.fit_transform(scaled_input)
        
        df_filled = pd.DataFrame(
            scaler.inverse_transform(scaled_output), columns=df_test.columns
        )
        df_filled.index = df_test.index
        diff = {"n": n}
        for col in cols_to_impute:
            index_to_fill = df_test.loc[df_test[col].isna()].index
            df_hist = pd.concat([
                df_hist,
                pd.DataFrame({
                    'n': [n] * len(index_to_fill),
                    'column': [col] * len(index_to_fill),
                    'diff': np.absolute(df_filled.iloc[index_to_fill][col].values - df_orig.iloc[index_to_fill][col].values),
                    'diff_relative': np.absolute(df_filled.iloc[index_to_fill][col].values - df_orig.iloc[index_to_fill][col].values) / np.mean(df_orig.iloc[index_to_fill][col].values),
                    'orig': df_orig.iloc[index_to_fill][col].values,
                    'imputed': df_filled.iloc[index_to_fill][col].values
                })
            ])
            df_filled.iloc[index_to_fill][col].values - df_orig.iloc[index_to_fill][col].values,
            df_orig.iloc[index_to_fill][col].values

    fig = go.Figure()
    col_pal = px.colors.qualitative.Dark24
    col_pal_iterator = itertools.cycle(col_pal)
    for col in cols_to_impute:
        new_colour = next(col_pal_iterator)
        loc_df = df_hist.loc[df_hist["column"]==col, ["n", "diff_relative"]].groupby("n").mean().reset_index()
        fig.add_trace(
            go.Scatter(
                x=loc_df.n,
                y=loc_df["diff_relative"],
                mode="lines",
                line=dict(color=new_colour, dash="solid"),
                name=f"moyenne diff {col}",
            )
        )
        loc_df = df_hist.loc[df_hist["column"]==col, ["n", "diff_relative"]].groupby("n").max().reset_index()
        fig.add_trace(
            go.Scatter(
                x=loc_df.n,
                y=loc_df["diff_relative"],
                mode="lines",
                line=dict(color=new_colour, dash="dash"),
                name=f"max diff {col}",
            )
        )
        loc_df = df_hist.loc[df_hist["column"]==col, ["n", "diff_relative"]].groupby("n").median().reset_index()
        fig.add_trace(
            go.Scatter(
                x=loc_df.n,
                y=loc_df["diff_relative"],
                mode="lines",
                line=dict(color=new_colour, dash="dot"),
                name=f"médiane diff {col}",
            )
        )

    fig.update_yaxes(
        type="log",
        title="Erreur/valeur imputée",
    )
    fig.update_xaxes(title="Valeur de k")
    fig.update_layout(
        title="Erreurs moyennes et max / KNNImputer",
        width=1500,
        height=900,
        template="plotly_white",
    )
    fig.show()

    if 1==0:
        for n in k_to_test:
            fig = go.Figure()
            col_pal_iterator = itertools.cycle(col_pal)
            for col in cols:
                new_colour = next(col_pal_iterator)
                fig.add_trace(
                    go.Histogram(
                        x=df_hist.loc[(df_hist["n"]==n) & (df_hist["column"]==col), "diff_relative"],
                        name=col,
                        marker_color=new_colour
                    )
                )
            # Overlay both histograms
            fig.update_layout(
                title=f"Répartition des erreurs pour k={n}",
                barmode='overlay'
            )
            # Reduce opacity to see both histograms
            fig.update_traces(opacity=0.75)
            fig.show()
    return df_hist


def display_knn_errors(df, n_neighbors=[1,2,3,4,5], sizes_to_test=None, col_to_predict=None, data_cols=None):
    if col_to_predict==None:
        raise Exception(
            "No column to predict"
        )

    if sizes_to_test is None:
        raise Exception(
            "No size to test"
        )

    if data_cols is None:
        data_cols = []
        for c in df.columns:
            if df.dtypes[c] != "O":
                data_cols.append(c)

    q = f""" `{col_to_predict}`.notnull() """
    for c in data_cols:
        q += f""" & `{c}`.notnull() """

    df_knn = df.query(q).copy(deep=True)
    df_knn.reset_index(drop=True, inplace=True)

    dict_knn_errors = {
        "size": [], 
        "n": [],
        "error": []
    }

    for s in sizes_to_test:
        sample_indexes = np.random.randint(df_knn.shape[0], size=s)

        data = df_knn.loc[sample_indexes, data_cols]
        target = df_knn.loc[sample_indexes, col_to_predict]

        xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.75)
        for n in n_neighbors:
            knn = neighbors.KNeighborsClassifier(n_neighbors=n)
            knn.fit(xtrain, ytrain)
            error = 1 - knn.score(xtest, ytest)
            dict_knn_errors["size"].append(s)
            dict_knn_errors["n"].append(n)
            dict_knn_errors["error"].append(error)

    df_knn_errors = pd.DataFrame(dict_knn_errors)

    fig = go.Figure()
    col_pal = px.colors.qualitative.Dark24
    col_pal_iterator = itertools.cycle(col_pal)
    for k in n_neighbors:
        fig.add_trace(
            go.Scatter(
                x=df_knn_errors.loc[df_knn_errors["n"]==k]["size"],
                y=df_knn_errors.loc[df_knn_errors["n"]==k]["error"],
                mode="lines",
                line=dict(color=next(col_pal_iterator), dash="solid"),
                name=f"knn avec k= {k}",
            )
        )
    fig.update_yaxes(
        title="Erreur",
    )
    fig.update_xaxes(
        title="Taille échantillon",
    )
    fig.update_layout(
        title=f"Erreurs KNN sur {col_to_predict} / k",
        width=1500,
        height=900,
        template="plotly_white",
    )
    fig.show()
