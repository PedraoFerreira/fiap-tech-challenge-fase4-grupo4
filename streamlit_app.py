import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.pipeline import Pipeline
from utils import DropFeatures, RenameFeatures, FilterDataSet, SetDateIndex
from joblib import load


df = pd.read_excel('dados/base_historica_consolidada.xlsx')

def pipeline(df):
    pipeline = Pipeline([
        ('feature_dropper', DropFeatures()),
        ('feature_renamer', RenameFeatures()),
        ('filter_dataset', FilterDataSet()),
        ('set_date_index', SetDateIndex()),
    ])
    df_pipeline = pipeline.fit_transform(df)
    return df_pipeline

df = pipeline(df)

# Dividir os dados em conjuntos de treinamento e teste
train_size = int(len(df) * 0.90)
train, test = df[:train_size], df[train_size:]

# Ajustar o modelo SARIMA aos dados de treinamento
order = (1, 1, 1)  # Ordem dos componentes ARIMA
seasonal_order = (1, 1, 1, 12)  # Ordem sazonal

# Gerar modelo de teste para calcular a accurancy 
model_acc = SARIMAX(train['preco'], order=order, seasonal_order=seasonal_order)
result_acc = model_acc.fit(disp=False)
start_index_acc = len(train)
end_index_acc = len(train) + len(test) - 1
predictions_acc = result_acc.predict(start=start_index_acc, end=end_index_acc, dynamic=False, typ='levels')

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(test['preco'], predictions_acc)

# Gera modelo com toda a base para prever o futuro realmente
model = SARIMAX(df['preco'], order=order, seasonal_order=seasonal_order)
result = model.fit(disp=False)
prev_meses = st.sidebar.number_input('Numero de meses futuros que deseja prever:', step=1, min_value=2, max_value=12)

# Fazer previsões
start_index = len(df)
end_index = len(df) + prev_meses - 1
predictions = result.predict(start=start_index, end=end_index, dynamic=False, typ='levels')

# Configuração do aplicativo Streamlit
st.title('Previsão do Preço do Barril de Petróleo')

# Plotar a série temporal modelo
st.subheader('Período Modelo:')

fig = plt.figure(figsize=(12, 6))
plt.plot(train.index, train['preco'], label='Treinamento')
plt.plot(test.index, test['preco'], label='Teste')
plt.plot(test.index, predictions_acc, label='Previsões Teste', linestyle='--')
plt.title('Previsões do Preço do Barril de Petróleo')
plt.xlabel('Data')
plt.ylabel('Preço do Barril de Petróleo')
plt.legend()
st.pyplot(fig)


st.write('Modelo utilizado:','SARIMAX')
st.write('Periodo utilizado para treino:',
        df.index.min(),
        ' à ',
        df.index.max()
)
st.write('MAPE:', round(mape,3))
st.write('Sendo assim, temos uma precisão de: ', round(100 - mape, 3))

# Plotar as previsões
st.subheader('Grafico de Previsões:')
st.line_chart(predictions, use_container_width=True, )

st.subheader('Valores Previstos:')
st.table(predictions)
