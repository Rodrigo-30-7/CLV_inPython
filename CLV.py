# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:29:54 2020

@author: FelipeHernandez
"""

# OTRA PRUEBAS #
from lifetimes.plotting import plot_frequency_recency_matrix, plot_probability_alive_matrix, plot_period_transactions
from lifetimes.utils import calibration_and_holdout_data
from lifetimes.datasets import load_transaction_data
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases
import matplotlib.pyplot as plt
from lifetimes.datasets import load_cdnow_summary #SE IMPORTAN NUEVAS FUNCIONES A UTLIZAR PARA PRUEBAS DEL MODELO

#filtered_data=data[['Columna1','Columna1']].drop_duplicates()
#orders[].describe()
#orders.info()

#os.getcwd()
#os.chdir("C:/Users/FelipeHernandez/Desktop")

filename = 'Guante_2018-2019.csv'
data_path = pd.read_csv(filename, header=0)


#print (data.head(10))
#data.head()

#data_path = "C:/Users/FelipeHernandez/Desktop/CDNOW_master-3v.txt" #SE LLAMA A LA BASE DE DATOS
#data_path = "C:/Users/FelipeHernandez/Desktop/fork_19-20.txt"
data_path = "C:/Users/FelipeHernandez/Desktop/nissan_18-19.txt" #SE LLAMA A LA BASE DE DATOS
#data_path = "C:/Users/FelipeHernandez/Desktop/guante_18-19.txt"
d = 'date'

orders_list = []
print(data_path)
with open(data_path) as f:
   for line in f:
        inner_list = [
            line.strip() #ELIMINA SALTOS DE LÍNEA EN EL ARCHIVO
            for line in re.split('\s+', line.strip()) #DIVIDE UN STRING POR ALGÚN TIPO DE PATRÓN
        ]
        orders_list.append(inner_list)


#orders = pd.DataFrame(orders_list, columns=['id', 'date', 'spend']) #ASIGNA LOS DATOS CARGADOS A COLUMNAS ORDENADAS DE UN DATAFRAME (PARA FORK)
orders = pd.DataFrame(orders_list, columns=['date', 'id', 'spend']) #ASIGNA LOS DATOS CARGADOS A COLUMNAS ORDENADAS DE UN DATAFRAME (PARA EL RESTO)

orders['date'] = pd.to_datetime(orders['date']) #SE ASEGURA DE GUARDAR LA VARIABLE 'DATE' EN FECHA
orders['spend'] = orders['spend'].astype(float) #SE ASEGURA DE GUARDAR LA VARIABLE 'SPEND' EN NÚMEROS REALES
orders = orders[orders['spend'] > 0] #CONDICIONA A TRABAJAR SOLO CON VALORES 'SPEND' SOBRE 0
#orders = orders[orders['id'] > 0] #X

orders.head() #MUESTRA EN PANTALLA LAS PRIMERAS N FILAS DEL ARCHIVO ACTUAL
orders.head()


print("########## ANALYTICS CLV ##########")
group_by_customer = orders.groupby(
    by = orders["id"],
    as_index = False) #AGRUPA POR ID
customers = group_by_customer["date"] \
    .agg(lambda x: (x.max() - x.min())) #AGRUPA POR FECHAS

customers["age"] = maximum(customers["date"] \
    .apply(lambda x: ceil(x.days / 30)), 1.0) #CALCULA LA EDAD DE VIDA DE UN CLIENTE
customers = customers.drop(columns = "date") #CORTA LA FECHA DE LA TABLA
customers.head() #MUESTRA EN PANTALLA LAS PRIMERAS N FILAS DEL ARCHIVO ACTUAL
print(customers) #MUESTRA EN PANTALLA TODAS LAS FILAS DEL ARCHIVO ACTUAL
time_to_churn = timedelta(weeks=8.6) #ASIGNA A UNA NUEVA VARIABLE MEDIANTE LA FUNCIÓN timdelta UNA CONDICIÓN DE TIEMPO (EN SEMANAS)
cutoff_date = orders['date'].max() #ASIGNA UNA NUEVA VARIABLE LA ÚLTIMA COMPRA DE UN CLIENTE (PERIODO DE TÉRMINO DE OBSERVACIÓN)

dead = group_by_customer['date'].max()['date'] \
    .apply(lambda x: (cutoff_date - x) > time_to_churn) #ESTIMA SI UN CLIENTE SE CONDIERA VIVO O MUERTO COMO COMPRADOR (ÚLTIMA FECHA DE COMPRA

churn = dead.sum() / customers['age'].sum() #CALCULA LA TASA DE FUGA MEDIANTE LA SUMA DE CLIENTES MUERTOS PARTIDOS POR LA SUMA DE SUS EDADES COMO CLIENTES
spend = orders['spend'].sum() / customers['age'].sum() #CALCULA LA TASA DE FUGA MEDIANTE LA SUMA DE MONTOS PARTIDOS POR LA SUMA DE SUS EDADES COMO CLIENTES

clv_aa = spend / churn #CALCULA EL CLV (ÚNICO PARA TODO) DESDE LA FÓRMULA TICKET PROMEDIO SOBRE TASA DE FUGA
print(clv_aa) #MUESTRA EL VALOR DEL CLV






print("########## PREDICTIVE CLV ##########")

      
#### CALCULAR RFM MÁS EDAD DEL CLIENTE ####
data = summary_data_from_transaction_data( #LA UTILIDAD SUMMARY[...] CALCULA FRECUENCIA, RECENCIA Y T (EDAD) DE CADA CLIENTE
    orders, 'id', 'date', 
    monetary_value_col='spend', #ASIGNA LA VARIABLE 'spend' (GASTOS TOTALES) A LA COLUMNA MONETARY
    observation_period_end = cutoff_date)
data.head()
print(data)


#data[data['frequency'] > 0]
#data[data['recency'] > 0]

data['frequency'].mean()
data['recency'].mean()
data['T'].mean()
data['monetary_value'].mean()



#### AJUSTE DE UN MODELO BETA GEOMÉTRICO A LOS DATOS PARA PREDECIR TRANSACCIONES FUTURAS ####
bgf = BetaGeoFitter(penalizer_coef=0.0) #SE CREA LA DISTRIBUCIÓN BETA GEOMÉTRICA CON COEFICIENTE DE PENALIZACIÓN=0 (LA CUAL DEPENDE DEL TAMAÑO DE MUESTRA)
bgf.fit(data['frequency'], data['recency'], data['T']) #SE AJUSTA ESTA DISTRIBUCIÓN A LA FRECUENCIA, GASTOS Y EDAD
future_horizon = 365 #SE ASIGNA UN VALOR A LA CONDICIÓN QUE SERÁ AGREGADA EN LA PREDICCIÓN DE TRANSACCIONES
data['predicted_purchases'] = bgf.predict(
    future_horizon,
    data['frequency'],
    data['recency'],
    data['T']) #SE REALIZA LA PREDICCIÓN DE TRANSACCIONES FUTURAS MEDIANTE LA FUNCIÓN 'predict'
# PROBABILIDAD DE COMPRA POR CLIENTE #
#t = 1
#print(bgf)
#bgf.summary

#plot_frequency_recency_matrix(bgf)
#plot_probability_alive_matrix(bgf)
plot_period_transactions(bgf) #COMPARATIVA DE FRECUENCIAS DE TRANSACCIONES
data.head()
print(data) #MUESTRA EN PANTALLA LAS PRIMERAS N FILAS DEL ARCHIVO ACTUAL


data['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(t, data['frequency'], data['recency'], data['T'])
data.sort_values(by='predicted_purchases').tail(10) #PREDICE UNA PROBABILIDAD 


#### SEPARAR BASE EN ENTRENAMIENTO Y PRUEBA ####
summary_cal_holdout = calibration_and_holdout_data(orders, 'id', 'date',
                                        calibration_period_end='20180830',
                                        observation_period_end='20190228' ) #SE SEPARA LA BASE DE DATOS EN PERIODOS DE ENTRENAMIENTO Y PRUEBA
print(summary_cal_holdout.head()) #MUESTRA LAS NUEVAS BASES DE DATOS


bgf.fit(summary_cal_holdout['frequency_cal'],
        summary_cal_holdout['recency_cal'],
        summary_cal_holdout['T_cal'])
plot_calibration_purchases_vs_holdout_purchases(bgf, summary_cal_holdout) #PRESENTA GRÁFICO DE COMPRAS REALES VS COMPRAS PREDICHAS






# #### AJUSTE DE UN MODELO GAMMA A LOS DATOS PARA PREDECIR EL TICKET PROMEDIO FUTURO ####
returning_customers_summary = data[data['frequency'] > 0] #TRANSFORMA A UNA NUEVA BASE DONDE SÓLO SE TRABAJARÁ CON CLIENTES QUE VUELVEN A COMPRAR

returning_customers_summary[['monetary_value', 'frequency']].corr() #SI LA CORRELACIÓN ES PEQUEÑA, EL MODELO ES ADUECUADO

ggf = GammaGammaFitter(penalizer_coef=0.001) #SE CREA UNA DISTRIBUCIÓN GAMMA CON COEFICIENTE DE PENALIZACIÓN=0 (LA CUAL DEPENDE DEL TAMAÑO DE MUESTRA)
ggf.fit(
    returning_customers_summary['frequency'],
    returning_customers_summary['monetary_value']) #SE AJUSTA ESTA DISTRIBUCIÓN A LA FRECUENCIA Y GASTOS
transaction_spend = ggf.conditional_expected_average_profit(
    data['frequency'],
    data['monetary_value'] #MEDIANTE LA FUNCION 'conditional[...] SE CALCULA LA EXPECTATIVA CONDICIONAL DEL BENEFICIO PROMEDIO POR TRANSACCIÓN.
).mean() 
print(transaction_spend) #MUESTRA EN PANTALLA EL TICKET PROMEDIO PREDICHO




#### CUSTOMER_AC DESDE EL MODELO 2 DE COHORTES ####
customers_ac = customers.merge(
    group_by_customer['spend'].sum(),
    on = 'id')
 
customers_ac['clv'] = customers_ac['spend'] / customers_ac['age'] / churn
customers_ac.head()
print(customers_ac)

#### CALCULAR CLV A NIVEL PREDICTIVO ####
customers_pm = customers_ac.join(
    data['predicted_purchases'],
    on='id',
    how='left'
).drop(columns='clv')

customers_pm['clv'] = customers_pm \
    .apply(
    lambda x: x['predicted_purchases'] * transaction_spend,
    axis=1) #CALCULA LA ÚLTIMA COLUMNA DE LA TABLA (CLV) EN BASE A LA MULTIPLICACIÓN DE LAS TRANSACCIONES Y LOS MONTOS PREDICHOS
customers_pm.tail() #MUESTRA LOS ÚLTIMOS VALORES DE LA TABLA FINAL CON LOS CÁLCULOS DE CLV PREDICTIVO
print(customers_pm) #MUESTRA LA TABLA FINAL CON LOS CÁLCULOS DE CLV PREDICTIVO

print(customers_pm)

customers_pm[customers_pm['clv']>0].mean()


customers_pm['age'].mean()
customers_pm['spend'].mean()
customers_pm['predicted_purchases'].mean()
customers_pm['clv'].mean()


#### MEDIDAS ####
media_t = data['T'].mean()
mediana_t = data['T'].median()
moda_t = data['T'].mode()
print("""
    Media: %d
    Mediana: %d
    Moda: %d
""" % (media_t,mediana_t,moda_t)) # GASTOS

media_fq = returning_customers_summary['frequency'].mean()
mediana_fq = returning_customers_summary['frequency'].median()
moda_fq = returning_customers_summary['frequency'].mode()
print("""
    Media: %d
    Mediana: %d
    Moda: %d
""" % (media_fq,mediana_fq,moda_fq)) # FRECUENCIA

media_mv = returning_customers_summary['monetary_value'].mean()
mediana_mv = returning_customers_summary['monetary_value'].median()
moda_mv = returning_customers_summary['monetary_value'].mode()
print("""
    Media: %d
    Mediana: %d
    Moda: %d
""" % (media_mv,mediana_mv,moda_mv)) # GASTOS




#### TUNNEAR MODELO BASADO EN RANDOMIZE SEARCH Y GRID SEARCH ####

#print(__doc__)

import numpy as np

from time import time
import scipy.stats as stats
from sklearn.utils.fixes import loguniform

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier

# get some data
#X, y = load_digits(return_X_y=True)

# build a classifier
#clf = SGDClassifier(loss='hinge', penalty='elasticnet',
#                    fit_intercept=True)


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {'average': [True, False],
              'l1_ratio': stats.uniform(0, 1),
              'alpha': loguniform(1e-4, 1e0)}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

# use a full grid over all parameters
param_grid = {'average': [True, False],
              'l1_ratio': np.linspace(0, 1, num=10),
              'alpha': np.power(10, np.arange(-4, 1, dtype=float))}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
