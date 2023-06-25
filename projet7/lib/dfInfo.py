import pandas as pd
import plotly.express as px

import seaborn as sns
import matplotlib.pyplot as plt

def _get_nb_filled(dataframe, column_name):
    return dataframe.loc[~dataframe[column_name].isna()].shape[0]

def _get_nb_unique_values(dataframe, column_name):
    return dataframe[column_name].unique().shape[0]

def get_df_info(df):
    '''Takes in a dataframe df, returns a new dataframe with information on the original dataframe columns coverage with the following columns: 
        - column: names of the original columns
        - dtype: types of the original columns
        - filled_nb: the number of filled cells (!=Naxx) for the specified column
        - filled_ratio: ratio nb filled cells / nb total cells
        - nb_unique_values: nb of different values in the original dataframe for the specified column

        Parameters:
            df (pandas.DataFrame): dataframe to analyze

        Returns:
            pandas.DataFrame
    '''
    df_info = pd.DataFrame({"column":list(df.columns), "dtype":list(df.dtypes)})
    df_info["filled_nb"] = df_info.apply(lambda x: _get_nb_filled(df, x["column"]), axis=1)
    df_info["filled_ratio"] = round(100 * df_info["filled_nb"] / df.shape[0], 2)
    df_info["nb_unique_values"] = df_info.apply(lambda x: _get_nb_unique_values(df, x["column"]), axis=1)
    return df_info;


def show_df_info_bar(df=None, df_info=None,width=1000):
    '''
    Shows a horizontal bar chart for a DataFrame. The bar chart show the columns with their nb of filled cells ratios, 
    inside the bars the nb of unique values for the corresponding column, and the color of the bar shows the type of the column

        Parameters:
            df (pandas.DataFrame): dataframe to show the analysis. get_df_info will be first called to analyze an analysis dataframe df_info
            df_info (pandas.DataFrame): analysis dataframe on which the bar chart will be built
    '''
    if df is None and df_info is None:
        return

    if df_info is None and df is not None:
        df_info = get_df_info(df)

    fig = px.bar(df_info, 
        x='filled_ratio', 
        y='column', 
        orientation='h', 
        color='dtype', 
        text='nb_unique_values',
        color_discrete_sequence=px.colors.qualitative.G10,
        title="Répartition des variables par nombre d'observations renseignées, par nombre de valeurs distinctes et par type",
        labels={
            "nb_unique_values": "Nb valeurs distinctes",
            "column": "Nom de la colonne",
            "filled_ratio": "Taux de remplissage",
            "dtype": "Type de la variable"
        },
    )
    fig.update_layout({
        'width': width,
        'height':  max(140, df_info.shape[0]*17) + 160,
        'yaxis_categoryorder': 'total ascending',
    })
    fig.update_yaxes(tickfont={'size': 9})

    fig.show()


def show_cols_repartition(df, cols):
    '''
    Shows in one graph the boxes for dataframe df and columns in cols array

        Parameters:
            df (pandas.DataFrame): dataframe
            cols (list): column list
    '''
    df_bp = df.loc[:, list(reversed(cols))]
    plt.figure(figsize=(20,len(cols) * 10/11))
    sns.boxplot(x="value", y="variable", data=pd.melt(df_bp), orient="h")
    plt.show()


