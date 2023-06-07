import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline


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
    def __init__(self):
        self.columns = []
        self.simple_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        #self.num_transformer = StandardScaler()

    def transform(self,X,y=None):
        print("transform numFeaturesImputerTransformer")
        return self.simple_imputer.transform(X.loc[:, self.columns])

    def fit(self, X, y=None):
        print("fit numFeaturesImputerTransformer")
        self.columns =  X.select_dtypes(include=["float64", "int32"]).columns
        self.simple_imputer.fit(X.loc[:, self.columns])
        print("fit numFeaturesImputerTransformer2")
        return self 
    

class catFeaturesImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = []
        self.simple_imputer = SimpleImputer(missing_values = np.nan, strategy='constant', fill_value='missing')

    def transform(self,X,y=None):
        print("transform catFeaturesImputerTransformer")
        return self.simple_imputer.transform(X.loc[:, self.columns])

    def fit(self, X, y=None):
        print("fit catFeaturesImputerTransformer")
        self.columns = X.select_dtypes(include=["object"]).columns
        self.simple_imputer.fit(X.loc[:, self.columns])
        return self 
    

class numFeaturesScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = []
        self.scaler = StandardScaler()

    def transform(self,X,y=None):
        print("transform numFeaturesScalerTransformer")
        return self.scaler.transform(X.loc[:, self.columns], y)

    def fit(self, X, y=None):
        print("fit numFeaturesImputerTransformer")
        self.columns =  X.select_dtypes(include=["float64", "int32"]).columns
        self.scaler.fit(X.loc[:, self.columns])
        print("fit numFeaturesImputerTransformer2")
        return self 
    

class catFeaturesEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = []
        self.encoder = OneHotEncoder()

    def transform(self,X,y=None):
        print("transform catFeaturesImputerTransformer")
        return self.encoder.transform(X.loc[:, self.columns], y)

    def fit(self, X, y=None):
        print("fit catFeaturesImputerTransformer")
        self.columns = X.select_dtypes(include=["object"]).columns
        self.encoder.fit(X.loc[:, self.columns])
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
    

