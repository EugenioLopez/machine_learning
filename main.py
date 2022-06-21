# -*- coding: utf-8 -*-
"""
Modulo principal
"""
# %%=============================================================================
# IMPORTO LAS LIBRERIAS
# =============================================================================
# import copy
import os

import numpy as np
import pandas as pd
import copy

from analista import Analista
from data_manager import DataManager

from tablero import Tablero

rut = r"C:\Users\elope\OneDrive\Escritorio"
os.chdir(path=rut)
np.seterr(divide="ignore")

# =============================================================================
# IMPORTO LA DATA
# =============================================================================
carpeta = rut + r"\data\titanic"
archivo_train = carpeta + r"\train.csv"
archivo_test = carpeta + r"\test.csv"

x_train = pd.pandas.read_csv(archivo_train, delimiter=",", index_col="PassengerId")
x_train["ESTADO"] = "train"

x_test = pd.pandas.read_csv(archivo_test, delimiter=",", index_col="PassengerId")
x_test["ESTADO"] = "test"

X = pd.concat([x_train, x_test], axis=0)
y = X.pop("Survived")

dm_final = DataManager(nombre="titanic", carpeta=carpeta)
dm_final.cargar()

X_train = dm_final.X.loc[list(X["ESTADO"] == "train")]
y_train = dm_final.y.loc[list(X["ESTADO"] == "train")]
dm_train = DataManager(nombre="titanic", carpeta=carpeta, X=X_train, y=y_train)

X_test = dm_final.X.loc[list(X["ESTADO"] == "test")]
dm_test = DataManager(nombre="titanic", carpeta=carpeta, X=X_test)

# =============================================================================
# IMPORTO LA DATA
# =============================================================================
tablero = Tablero(nombre='tablero_titanic', carpeta=carpeta)
tablero.cargar()

tablero.actualizar_tablero(nombre_analistas='analista_titanic')
tablero.guardar()







# Creo el analista y veo las observaciones
self = tablero.crear_analista(nombre_analistas='analista_titanic', datamanager=dm_train)

# %%===========================================================================
# SPLITEO Y ESCALO LA SERIE
# =============================================================================
scalers = [
    "MinMaxScaler",
    "StandardScaler",
    "QuantileTransformer",
    "RobustScaler",
    "PowerTransformer"
]

def generar_analistas_split_scaler(scaler, guardar):

    self2 = copy.deepcopy(self)

    self2.analizar(
        splitear=True,
        funcion_split_train_test='train_test_split',
        parametros_split_train_test=dict(test_size=0.3, shuffle=False, random_state=0),
        funcion_split_test_demo='train_test_split',
        parametros_split_test_demo=dict(test_size=0.5, shuffle=False, random_state=0),
        scalar=True,
        scaler=scaler,
        ajustar=False,
    )

    self2.analizar_dataset_por_manifold(dataset='X_train', inplace=True)

    if guardar:
        self2.marcar_nueva_version()
        self.guardar_como()

    return self2


analistas = [generar_analistas_split_scaler(scaler, guardar=True) for scaler in scalers]
tablero.actualizar_tablero(nombre_analistas='analista_titanic')
tablero.guardar()

self = analistas[0]

# %%=============================================================================
# FEATURE SELECTION
# ===============================================================================
# Feature importance
self.calcular_features_importance(
    funcion_estimador='RandomForestClassifier',
    lista_param_grid=None,
)

# VarianceThreshold
self.calcular_feature_selection(
    nombre_funcion_modelo='VarianceThreshold',
    nombre_libreria_origen='sklearn.feature_selection',
    params_modelo=dict(threshold=0.005),
)

# LassoCV
self.calcular_feature_selection(
    nombre_funcion_modelo='LassoCV',
    nombre_libreria_origen='sklearn.linear_model',
    params_modelo=dict(cv=10, random_state=0),
)

# PCA
self.calcular_feature_extraction(
    nombre_funcion_modelo='PCA',
    nombre_libreria_origen='sklearn.decomposition',
    params_modelo=None,
)

# RFECV (GradientBoostingClassifier)
lista_estimadores = list()
# lista_estimadores.append("LGBMClassifier")
lista_estimadores.append('GradientBoostingClassifier')
# lista_estimadores.append('XGBClassifier')
# lista_estimadores.append('CatBoostClassifier')
# lista_estimadores.append('LogisticRegression')
# lista_estimadores.append('RandomForestClassifier')
self.calcular_feature_selection(
    nombre_funcion_modelo = 'RFECV',
    nombre_libreria_origen = 'sklearn.feature_selection',
    params_modelo=dict(
        estimator=self.generar_modelo('GradientBoostingClassifier', 'sklearn.ensemble'),
        step=1,
        cv=5,
        scoring="accuracy",
        min_features_to_select=1,
    )
)
self.analizar_dataset_por_manifold(dataset='X_train', inplace=True)
self.analisis_manifold['fig'].show()
self.guardar()



# GARS
self.calcular_feature_selection(
    nombre_funcion_modelo = 'GARS',
    nombre_libreria_origen = None,
    params_modelo = dict(
        cant_max_columnas=50,
        proba_apagar=0.2,
        parametros_algo=None,
    )
)

# %%===========================================================================
# CALCULO EL FEATURE IMPORTANCE
# =============================================================================
self = tablero.cargar_analista(nombre_analistas='analista_titanic', ind=0)

lista_estimadores = [
    'LGBMClassifier',
    'GradientBoostingClassifier',
    'RandomForestClassifier',
    'XGBClassifier',
    'CatBoostClassifier',
]

for estimador in lista_estimadores:

    self2 = self.estimar_por_funcion_estimador(
        funcion_estimador=estimador,
        lista_param_grid=None,
    )
    self2.marcar_nueva_version()
    self2.guardar_como()

# %%===========================================================================
# BUSCO LA MEJOR COMBINACION DE PARAMETROS POR ALGORITMO GENETICO
# =============================================================================
self = tablero.cargar_analista(nombre_analistas='analista_titanic', ind=0)


# Genero los parametros
lista_estimadores = list()
# lista_estimadores.append("LGBMClassifier")
# lista_estimadores.append('GradientBoostingClassifier')
# lista_estimadores.append('XGBClassifier')
# lista_estimadores.append('CatBoostClassifier')
lista_estimadores.append('LogisticRegression')
# lista_estimadores.append('RandomForestClassifier')


self.ajustar_por_algoritmo_genetico(
    lista_estimadores=lista_estimadores,
    agregar_hiperparametros=True,
    agregar_total_hiperparametros=True,  # <---- TODOS LOS PARAMETROS
    filtrar_scaler=False,  # <---- EL SCALER
    filtrar_columnas=False,  # True <---- LAS COLUMNAS
    probabilidad_ocurrencia_columna=0.5,
    filtrar_filas=False,  # <---- LAS FILAS
    probabilidad_ocurrencia_fila=0.5,
    dd_parametros_adicionales=None,
    parametros_algo=dict(
        n_population=50,
        n_generations=200,
        crossover_proba=1.0,
        mutation_proba=0.75,
        crossover_independent_proba=[0.05, 0.5],
        mutation_independent_proba=[0.05, 0.5],
        tournament_size=5,
        n_random_population_each_gen=0,
        add_mutated_hall_of_fame=True,
        n_gen_no_change=20,
    ),
)

self.marcar_nueva_version()
self.guardar_como()

d = self.predecir_por_individuo(
    individuo=self.mejor_individuo_algoritmo_genetico,
    usar_train=True,
    usar_test=True,
    usar_demo=True,
    calcular_proba=True,
    )

self2 = copy.deepcopy(self)
self2.estimar_por_individuo(
    individuo=self.mejor_individuo_algoritmo_genetico,
)

self.mejorar_metricas_algoritmo_genetico()
self.guardar()
a = pd.DataFrame.from_dict(self.lista_individuos)

# %%===========================================================================
# EVALUO POR VOTING
# =============================================================================
res_voting = self.predecir_por_voting(
    d_filtros=[["metrica_minima", "mayor", 0.81]],
    funcion_scorer='accuracy_score',
    parametros_scorer=dict(),
    usar_train=True,
    usar_test=True,
    usar_demo=True,
    corte=0.5,
    voting='soft',
)

# %%===========================================================================
# AGREGO EL TEST Y EVALUO POR VOTING
# =============================================================================
self = Analista(nombre="analista_titanic", carpeta=carpeta)
self = self.cargar_versiones(ind=-1)[0]

# Tomo el X e y
X = self.datamanager.X
y = self.datamanager.y
X_true = X.loc[y.isnull()]
y_true = pd.Series(np.ones(X_true.shape[0]))

# Agrego el test
self.agregar_test(X_true=X_true, y_true=y_true, usar_scaler=True)

# Calculo el voting nuevamente con el nuevo test
res_voting = self.predecir_por_voting(
    d_filtros=[["metrica_minima", "mayor", 0.807]],
    funcion_scorer='accuracy_score',
    parametros_scorer=dict(),
    usar_train=True,
    usar_test=True,
    usar_demo=True,
    corte=0.5,
    voting='soft',
)

# Tomo el predict que necesito
y_predict_submission = res_voting['y_predict_test']

# %%=============================================================================
# GENERO EL ARCHIVO SUBMISSION
# =============================================================================
self.generar_submission(
    X_true=None,
    y_true=y_predict_submission,
    exportar=True,
    nombres_columnas=["PassengerId", "Survived"],
    nombre_archivo="gender_submission",
)


self2 = copy.deepcopy(self)
self2.agregar_test(X_true=X_test, y_true=None, usar_scaler=True)
X_test_sc = self2.X_test

self2 = copy.deepcopy(self)
self2.agregar_test(X_true=self.X, y_true=None, usar_scaler=True)
X_test_sc = self2.X_test

registro = [list(X_test_sc.iloc[0, :])]
registro2 = self.X_train.values




from sklearn.metrics.pairwise import euclidean_distances

distances = euclidean_distances(registro, registro2)
index_cercano = pd.Series(distances[0]).sort_values().head(1).index[0]