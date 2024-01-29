import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder

FEATURES_TO_DROP = [
    'Salário mínimo real - R$',
    'PIB - R$ (milhões)',
    'Taxa de câmbio - R$ / US$ - comercial - compra - média',
    'INPC - geral - índice'
]

FEATURES_TO_RENAME = {
    'Média de Preço - petróleo bruto - Brent (FOB)': 'preco',
    'Mês/Ano' : 'data'
}


# Classes para pipeline
class DropFeatures(BaseEstimator,TransformerMixin):
    def __init__(self,feature_to_drop = FEATURES_TO_DROP):
        self.feature_to_drop = feature_to_drop
    
    def fit(self,df):
        return self
    
    def transform(self,df):
        if (set(self.feature_to_drop).issubset(df.columns)):
            df.drop(self.feature_to_drop,axis=1,inplace=True)
            return df
        else:
            print('Uma ou mais features não estão no DataFrame')
            return df

class RenameFeatures(BaseEstimator,TransformerMixin):
    def __init__(self,features_to_rename = FEATURES_TO_RENAME):
        self.features_to_rename = features_to_rename

    def fit(self,df):
        return self

    def transform(self,df):
        if (set(self.features_to_rename.keys()).issubset(df.columns)):
            df.rename(columns=self.features_to_rename, inplace=True)
            return df
        else:
            print('Uma ou mais colunas não estão no DataFrame')
            return df

class FilterDataSet(BaseEstimator,TransformerMixin):
    def __init__(self,data_column_name = 'data', gte_date = '2013-01-01'):
        self.data_column_name = data_column_name
        self.gte_date = gte_date
    def fit(self,df):
        return self
    def transform(self,df):
        if self.data_column_name in df.columns:
            df[self.data_column_name] = pd.to_datetime(df[self.data_column_name])
            df = df[df[self.data_column_name] >= self.gte_date]
            return df
        else:
            print(f'Coluna {self.index_column_name} não está no DataFrame')
            return df


class SetDateIndex(BaseEstimator,TransformerMixin):
    def __init__(self,index_column_name = 'data'):
        self.index_column_name = index_column_name
    def fit(self,df):
        return self
    def transform(self,df):
        if self.index_column_name in df.columns:
            df.set_index(self.index_column_name, inplace=True)
            return df
        else:
            print(f'Coluna {self.index_column_name} não está no DataFrame')
            return df