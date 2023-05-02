from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline

from collections import Counter
from dateutil import parser

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import NotFittedError

class featureCopyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,copy_col_from, copy_col_to):
        self.copy_col_from=copy_col_from
        self.copy_col_to=copy_col_to

    def transform(self,X,y=None):
        X[self.copy_col_to] = X[self.copy_col_from].values
        return X

    def fit(self, X, y=None):
        return self 


class featureSubstractTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,substract_from, substract_col, result_col):
        self.substract_from=substract_from
        self.substract_col=substract_col
        self.result_col=result_col

    def transform(self,X,y=None):
        X[self.result_col] = X[self.substract_from] - X[self.substract_col]
        return X

    def fit(self, X, y=None):
        return self 


class featureDropTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,columns):
        self.columns=columns

    def transform(self,X,y=None):
        for c in self.columns:
            if c in X.columns:
                X.drop(columns=[c], inplace=True)
        return X

    def fit(self, X, y=None):
        return self 


class featureDropAllButTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,keep_exact_columns=[], keep_columns_starting_with=[] ):
        self.keep_exact_columns=keep_exact_columns
        self.keep_columns_starting_with=keep_columns_starting_with
        self.drop_columns=[] 

    def transform(self,X,y=None):
        X = X.drop(columns=self.drop_columns)
        return X

    def fit(self, X, y=None):
        for c in X.columns:
            if c not in self.keep_exact_columns:
                s = False
                for k in self.keep_columns_starting_with:
                    if c.startswith(k):
                        s = True
                        break
                if s==False:
                    self.drop_columns.append(c)
        return self 


class featureLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.X_columns_to_logify = []
        
    def transform(self,X,y=None):
        for c in self.X_columns_to_logify:
            X[c] = np.log1p(X[c])
        return X

    def fit(self, X, y=None):
        for c in X.select_dtypes(include=["float64", "int32"]).columns:
            if X[c].kurtosis()>8 or abs(X[c].skew())>1:
                self.X_columns_to_logify.append(c)
        return self 

    
class featureFloatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,columns):
        self.columns=columns

    def transform(self,X,y=None):
        for c in self.columns:
            if c in X.columns:
                X[c] = X[c].astype(float)
        return X

    def fit(self, X, y=None):
        return self 

    
class featureDateTimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,columns):
        self.columns=columns

    def transform(self,X,y=None):
        for c in self.columns:
            if c in X.columns:
                X[c] = X[c].astype('datetime64[ns]')
        return X

    def fit(self, X, y=None):
        return self 

class featureSplitterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, from_column, splitter_fn=lambda x: x, split_char_1=",", split_char_2=":", prefix=""):
        print(f"featureSplitterTransformer, from_column={from_column}")
        self.from_column = from_column
        self.split_char_1 = split_char_1
        self.split_char_2 = split_char_2
        self.splitter_fn = splitter_fn
        self.prefix = prefix
        self.new_cols = []
        
    def create_new_col_names(self, X):
        new_cols = []
        print(self.from_column)
        for g in X[self.from_column]:
            if g is None:
                g = ""
            for g2 in g.split(self.split_char_1):
                c = g2.split(self.split_char_2)
                k = self.splitter_fn(c[0], self.prefix)
                if k not in X.keys():
                    new_cols.append(k)
        return new_cols
        
    def apply_to_rows(self, x):
        s = x[self.from_column]
        if s is None:
            s = ""
        for g in s.split(self.split_char_1):
            c = g.split(self.split_char_2)
            if len(c)<2:
                #print(f"**** self.from_column={self.from_column}")
                #print(f"**** x={x}")
                #print(f"**** c={c}")
                continue
            k = self.splitter_fn(c[0], self.prefix)
            if k in x.keys():
                x[k] += float(c[1])
        return x
    
    def transform(self, X, y=None):
        for nc in self.new_cols:
            if nc not in X.columns:
                X[nc] = np.zeros(X.shape[0])
        return X.apply(lambda x: self.apply_to_rows(x), axis=1)
        
    def fit(self, X, y=None):
        self.new_cols = self.create_new_col_names(X)
        return self 
        
        
class mostPurchasedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, from_columns_prefix=[], to_col=None):
        self.from_columns_prefix = from_columns_prefix
        self.to_col = to_col
        
    def apply_to_rows(self, x, cols):
        most_purchased = None
        nb_most_purchased = 0
        for c in cols:
            if c.startswith(self.from_columns_prefix):
                if x[c]>0 and (most_purchased is None or x[c]>nb_most_purchased):
                    most_purchased = c
                    nb_most_purchased = x[c]
        return most_purchased
    
    def transform(self, X, y=None):
        if self.to_col not in X.columns:
            X[self.to_col] = np.NaN
        #print(f"X[{self.to_col}].shape={X[self.to_col].shape}")
        r = X.apply(lambda x: self.apply_to_rows(x, X.columns), axis=1)
        #print(f"r.shape={r.shape}")
        X[self.to_col] = r
        return X
        
    def fit(self, X, y=None):
        return self 
        
        
class featureMultiplierTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, from_column, factor=1):
        self.column = column
        self.from_column = from_column
        self.factor = factor

    def transform(self,X,y=None):
        X.loc[:, self.column] = X[self.from_column] * self.factor
        return X

    def fit(self, X, y=None):
        return self 
    
class weekdayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,date_column, weekday_column):
        self.date_column=date_column
        self.weekday_column = weekday_column

    def transform(self,X,y=None):
        X[self.weekday_column] = X[self.date_column].dt.dayofweek
        return X

    def fit(self, X, y=None):
        return self 
    

class mostCommonWeekdayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,dates_column, weekday_column):
        self.dates_column=dates_column
        self.weekday_column = weekday_column

    def apply_to_rows(self, x):
        t = x[self.dates_column].split(",")
        t2 = []
        for t0 in t:
            t2.append(int(parser.parse(t0).strftime('%w')))
        r = Counter(t2).most_common()[0][0]-1
        if r == -1:
            r = 6
        x[self.weekday_column] = r
        return x

    def transform(self,X,y=None):
        X[self.weekday_column] = np.zeros(X.shape[0])
        return X.apply(lambda x: self.apply_to_rows(x), axis=1)

    def fit(self, X, y=None):
        return self 
    
class minValueTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,dates_column, first_date_column):
        self.dates_column=dates_column
        self.first_date_column = first_date_column

    def apply_to_rows(self, x):
        t = x[self.dates_column].split(",")
        x[self.first_date_column] = min(t) if len(t)>0 else np.nan
        return x

    def transform(self,X,y=None):
        X[self.first_date_column] = np.NaN
        return X.apply(lambda x: self.apply_to_rows(x), axis=1)

    def fit(self, X, y=None):
        return self 
    
class dateAgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,column_date, target_column, ref_dt=None):
        self.column_date=column_date
        self.target_column=target_column
        if ref_dt is None:
            self.ref_dt = np.datetime64('now') 
        else:
            self.ref_dt = ref_dt

    def transform(self,X,y=None):
        d = (self.ref_dt - pd.to_datetime(X[self.column_date]))
        X[self.target_column] = d.dt.days
        return X

    def fit(self, X, y=None):
        return self 


class renameColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,from_name, to_name):
        self.from_name=from_name
        self.to_name=to_name

    def transform(self,X,y=None):
        return X.rename(columns={self.from_name: self.to_name})

    def fit(self, X, y=None):
        return self 


class upperTextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,columns):
        self.columns=columns

    def transform(self,X,y=None):
        for c in self.columns:
            X[c] = X[c].str.upper()
        return X

    def fit(self, X, y=None):
        return self 
    

class valueGroupTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_columns, target_values):
        self.target_columns = target_columns
        self.target_values = target_values

    def transform(self,X,y=None):
        for i in range(0, len(self.target_columns)):
            c = self.target_columns[i]
            for j in range(0, len(self.target_values[i])):
                v = self.target_values[i][j].upper()
                X.loc[X[c].str.upper().str.contains(v), c] = v
        return X

    def fit(self, X, y=None):
        return self 
    

class replaceNaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, default_value):
        self.columns = columns
        self.default_value = default_value

    def transform(self,X,y=None):
        for c in self.columns:
            X[c] = X[c].fillna(self.default_value)
        return X

    def fit(self, X, y=None):
        return self 
    

class medianImputationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_fix, ref_cols_groups, debug=False):
        self.cols_to_fix = cols_to_fix
        self.ref_cols_groups = ref_cols_groups
        self._X_ref = None
        self.debug = debug

    def debug_msg(self, msg):
        if self.debug:
            print(msg)
        
    def transform(self,X,y=None):
        self.debug_msg(f"----------- in medianImputationTransformer:transform, self._X_ref={self._X_ref.head(0)}")
        for col_to_fix in self.cols_to_fix:
            self.debug_msg(f"---------------------- in medianImputationTransformer:transform, col_to_fix={col_to_fix}")
            for ind in X.loc[X[col_to_fix].isna()].index:
                for ref_cols in self.ref_cols_groups:

                    X_filtered = self._X_ref.copy()

                    if col_to_fix not in X_filtered.columns:
                        self.debug_msg(f"""Error : Could not find column {col_to_fix} in columns {X_filtered.columns}""")

                    X_filtered = X_filtered.loc[~X_filtered[col_to_fix].isna()]
                    for c in ref_cols:
                        v = X.at[ind, c]
                        X_filtered = X_filtered.loc[~X_filtered[c].isna() & (X_filtered[c]==v)]

                    if X_filtered.shape[0]==0:
                        continue


                    if X[col_to_fix].dtype == np.dtype('O'):
                        X.at[ind, col_to_fix] = X_filtered[col_to_fix].value_counts().index[0]
                    else:
                        X.at[ind, col_to_fix] = X_filtered[col_to_fix].median()
        self.debug_msg(f"----------- end of medianImputationTransformer:transform, self._X_ref={self._X_ref.head(0)}")

        return X

    def fit(self, X, y=None):
        self.debug_msg(f"***************************in medianImputationTransformer:fit columns={X.columns}")
        self._X_ref = X.copy()
        return self 
    

class gfaFixTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, total_gfa_column, gfa_columns):
        self.total_gfa_column = total_gfa_column
        self.gfa_columns = gfa_columns

    def transform(self,X,y=None):
        q = f"""`{self.total_gfa_column}` < ( """
        for i in range(0, len(self.gfa_columns)):
            if i>0:
                q += " + "
            q += f"""`{self.gfa_columns[i]}`"""
        q += " )"
        
        for ref_index in X.query(q).index:
            total = 0
            for c in self.gfa_columns:
                total += X.at[ref_index,c]
            # On va considérer que total_gfa_column est la colonne avec la bonne valeur (pas d'imputation faite à ce niveau)
            # on corrige donc les colonnes de gfa_columns en utilisant le ratio 
            if X.at[ref_index, self.total_gfa_column]:
                # On évite les divisions par 0
                ratio = 1
            else:
                ratio = total /  X.at[ref_index, self.total_gfa_column]
            for c in self.gfa_columns:
                X.at[ref_index,c] = X.at[ref_index,c] / ratio
                
        return X

    def fit(self, X, y=None):
        return self 
    

class numRangeCatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, new_column, column, bin_size):
        self.new_column = new_column
        self.column = column
        self.bin_size = bin_size

    def transform(self,X,y=None):
        X[self.new_column] = np.char.mod("%d", np.rint(X[self.column] / self.bin_size) * self.bin_size)
        return X

    def fit(self, X, y=None):
        return self 
    

class numFeatureBounderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bounds):
        self.bounds = bounds

    def transform(self,X,y=None):
        for bound in self.bounds:
            if len(bound)>1:
                if bound[1] is not None:
                    X.loc[X[bound[0]]<bound[1], bound[0]] = np.nan
            if len(bound)>2:
                if bound[2] is not None:
                    X.loc[X[bound[0]]>bound[2], bound[0]] = np.nan
                    
        return X

    def fit(self, X, y=None):
        return self 
    

class ratioMakerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, combinations):
        self.combinations = combinations

    def transform(self,X,y=None):
        for combi in self.combinations:
            X[combi[0]] = 1
            X.loc[X[combi[2]]==0, combi[0]] = np.nan
            X.loc[X[combi[2]]!=0, combi[0]] = np.divide(X[combi[1]], X[combi[2]])
        return X

    def fit(self, X, y=None):
        return self 
    

class featureMultiplierImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, from_column, factor=1):
        self.column = column
        self.from_column = from_column
        self.factor = factor

    def transform(self,X,y=None):
        X.loc[X[self.column].isna() & ~X[self.from_column].isna(), self.column] = X[self.from_column] * self.factor
        return X

    def fit(self, X, y=None):
        return self 
    

class featureAdderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, columns_to_add):
        self.column = column
        self.columns_to_add = columns_to_add

    def transform(self,X,y=None):
        sum = X[self.columns_to_add[0]]
        for i in range(1, len(self.columns_to_add)):
            sum += X[self.columns_to_add[i]]
        X.loc[X[self.column].isna(), self.column] = sum
        return X

    def fit(self, X, y=None):
        return self 
    

class numFeaturesImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, imputer=None):
        self.columns = []
        if imputer is None:
            self.imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        else:
            self.imputer = imputer

    def transform(self,X,y=None):
        imputed = self.imputer.transform(X[self.columns])
        r = pd.DataFrame(imputed, columns = self.columns)
        for c in r.columns:
            X[c] = r[c]
        return X

    def fit(self, X, y=None):
        self.columns =  X.select_dtypes(include=["float64", "int64"]).columns
        self.imputer.fit(X.loc[:, self.columns])
        return self 
    

class catFeaturesImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, imputer=None):
        self.columns = []
        if imputer is None:
            self.imputer = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
        else:
            self.imputer = imputer
            
    def transform(self,X,y=None):
        imputed = self.imputer.transform(X[self.columns])
        r = pd.DataFrame(imputed, columns = self.columns)
        for c in r.columns:
            X[c] = r[c]
        return X

    def fit(self, X, y=None):
        self.columns = X.select_dtypes(include=["object"]).columns
        self.imputer.fit(X.loc[:, self.columns])
        return self 
    

class catFeaturesNone2NaNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[]):
        self.columns = columns
            
    def transform(self,X,y=None):
        for c in self.columns:
            X.loc[X[c].isna(), c] = np.NaN
        return X

    def fit(self, X, y=None):
        return self 
    

class numFeaturesScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, scaler):
        self.columns = []
        self.scaler = scaler

    def transform(self,X,y=None):
        print("transform numFeaturesScalerTransformer")
        print(X)
        return pd.DataFrame(self.scaler.transform(X.loc[:, self.columns]), columns = X.columns)

    def fit(self, X, y=None):
        print("fit numFeaturesImputerTransformer")
        self.columns =  X.select_dtypes(include=[np.number]).columns
        print(f"fitted columns={self.columns}")
        self.scaler.fit(X.loc[:, self.columns])
        print("fit numFeaturesImputerTransformer2")
        return self 
    

class catFeaturesEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, encoder):
        self.columns = []
        self.encoder = encoder

    def transform(self,X,y=None):

        #One-hot-encode the categorical columns.
        #Unfortunately outputs an array instead of dataframe.
        print(X.loc[:, self.columns])
        array_hot_encoded = self.encoder.transform(X.loc[:, self.columns])
        print("--------------------------------------------")
        print(array_hot_encoded)
        print("********************************************")
        #Convert it to df
        print(self.encoder.get_feature_names_out(self.columns))
        data_hot_encoded = pd.DataFrame([array_hot_encoded], columns=self.encoder.get_feature_names_out(), index=X.index)

        #Extract only the columns that didnt need to be encoded
        data_other_cols = X.drop(columns=self.columns)

        #Concatenate the two dataframes : 
        data_out = pd.concat([data_other_cols, data_hot_encoded], axis=1)
        
        
        print("transform catFeaturesImputerTransformer")
        return data_out

    def fit(self, X, y=None):
        print("fit catFeaturesImputerTransformer")
        self.columns = X.select_dtypes(include=["object"]).columns
        self.encoder.fit(X.loc[:, self.columns])
        print(f"columns to encode: {self.columns}")
        return self 
    

class queryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, q):
        self.q = q
        return
    
    def transform(self,X,y=None):
        print(X.query(self.q).head())
        return X

    def fit(self, X, y=None):
        return self 
    

class debugTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, s="DEBUG :", expr=None):
        self.s = s
        self.expr = expr
        return
    
    def transform(self,X,y=None):
        if self.expr is None:
            print(self.s)
        else:
            print(f"{self.s} {eval(self.expr)}")
        return X

    def fit(self, X, y=None):
        return self 
    

class DataFrameOneHotEncoder(BaseEstimator, TransformerMixin):
    """Specialized version of OneHotEncoder that plays nice with pandas DataFrames and
    will automatically set the feature/column names after fit/transform
    """

    def __init__(
        self,
        categories="auto",
        drop=None,
        sparse=None,
        dtype=np.float64,
        handle_unknown="error",
        col_overrule_params={},
    ):
        """Create DataFrameOneHotEncoder that can be fitted to and transform dataframes
        and that will set up the column/feature names automatically to
        original_column_name[categorical_value]

        If you provide the same arguments as you would for the sklearn 
        OneHotEncoder, these parameters will apply for all of the columns. If you want
        to have specific overrides for some of the columns, provide these in the dict
        argument col_overrule_params.
        
        For example:
            DataFrameOneHotEncoder(col_overrule_params={"col2":{"drop":"first"}})

        will create a OneHotEncoder for each of the columns with default values, but
        uses a drop=first argument for columns with the name col2

        Args:
            categories‘auto’ or a list of array-like, default=’auto’
                ‘auto’ : Determine categories automatically from the training data.
                list : categories[i] holds the categories expected in the ith column.
                The passed categories should not mix strings and numeric values
                within a single feature, and should be sorted in case of numeric
                values.
            drop: {‘first’, ‘if_binary’} or a array-like of shape (n_features,),
                default=None
                See OneHotEncoder documentation
            sparse: Ignored, since we always will work with dense dataframes
            dtype: number type, default=float
                Desired dtype of output.
            handle_unknown: {‘error’, ‘ignore’}, default=’error’
                Whether to raise an error or ignore if an unknown categorical feature
                is present during transform (default is to raise). When this parameter
                is set to ‘ignore’ and an unknown category is encountered during
                transform, the resulting one-hot encoded columns for this feature will
                be all zeros. In the inverse transform, an unknown category will be
                denoted as None.
            col_overrule_params: dict of {column_name: dict_params} where dict_params
                are exactly the options cateogires,drop,sparse,dtype,handle_unknown.
                For the column given by the key, these values will overrule the default
                parameters
        """
        self.categories = categories
        self.drop = drop
        self.sparse = sparse
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.col_overrule_params = col_overrule_params
        pass

    def fit(self, X, y=None):
        """Fit a separate OneHotEncoder for each of the columns in the dataframe

        Args:
            X: dataframe
            y: None, ignored. This parameter exists only for compatibility with
                Pipeline

        Returns
            self

        Raises
            TypeError if X is not of type DataFrame
        """
        if type(X) != pd.DataFrame:
            raise TypeError(f"X should be of type dataframe, not {type(X)}")

        self.onehotencoders_ = []
        self.column_names_ = []

        for c in X.columns:
            # Construct the OHE parameters using the arguments
            ohe_params = {
                "categories": self.categories,
                "drop": self.drop,
                "sparse_output": False,
                "dtype": self.dtype,
                "handle_unknown": self.handle_unknown,
            }
            # and update it with potential overrule parameters for the current column
            ohe_params.update(self.col_overrule_params.get(c, {}))

            # Regardless of how we got the parameters, make sure we always set the
            # sparsity to False
            ohe_params["sparse_output"] = False

            # Now create, fit, and store the onehotencoder for current column c
            ohe = OneHotEncoder(**ohe_params)
            self.onehotencoders_.append(ohe.fit(X.loc[:, [c]]))

            # Get the feature names and replace each x0_ with empty and after that
            # surround the categorical value with [] and prefix it with the original
            # column name
            feature_names = ohe.get_feature_names_out()
            feature_names = [x.replace("x0_", "") for x in feature_names]
            feature_names = [f"{c}[{x}]" for x in feature_names]

            self.column_names_.append(feature_names)

        return self

    def transform(self, X):
        """Transform X using the one-hot-encoding per column

        Args:
            X: Dataframe that is to be one hot encoded

        Returns:
            Dataframe with onehotencoded data

        Raises
            NotFittedError if the transformer is not yet fitted
            TypeError if X is not of type DataFrame
        """
        if type(X) != pd.DataFrame:
            raise TypeError(f"X should be of type dataframe, not {type(X)}")

        if not hasattr(self, "onehotencoders_"):
            raise NotFittedError(f"{type(self).__name__} is not fitted")

        all_df = []

        for i, c in enumerate(X.columns):
            ohe = self.onehotencoders_[i]

            transformed_col = ohe.transform(X.loc[:, [c]])

            df_col = pd.DataFrame(transformed_col, columns=self.column_names_[i])
            all_df.append(df_col)

        return pd.concat(all_df, axis=1)