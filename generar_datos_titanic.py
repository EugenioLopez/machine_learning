import os

import numpy as np
import pandas as pd

from data_manager import DataManager


rut = r"C:\Users\elope\OneDrive\Escritorio\disco viejo\lllibs\machine learning"
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

# self = dm_inicial

# %%===========================================================================
# PROCESO LOS DATOS
# =============================================================================
dm_inicial = DataManager(nombre="titanic", carpeta=carpeta, X=X, y=y)

# COMPLETO LOS NANS
dm_inicial.fillna(
    valor="mice",
    d_filtros=[["porcentaje_nulls", "menor", 0.3]],
    cantidad_datasets=5,
    verbose=False,
)
dm_inicial.fillna(
    valor=-1,
    d_filtros=[["porcentaje_nulls", "mayor o igual", 0.3]],
    cantidad_datasets=5,
    verbose=False,
)

# # GENERO FEATURES NUEVOS
df_x_creadas = pd.DataFrame()

# Name
df_x_creadas["Surname"] = dm_inicial.X["Name"].str.rsplit(", ", expand=True)[0]

# Cantidad de apellido
_sur_cant = dm_inicial.contar_repeticiones(df=df_x_creadas[["Surname"]])
df_x_creadas["Surname_cantidad"] = _sur_cant

# Edad
df_x_creadas["Age"] = dm_inicial.X["Age"]

# Limpio nombres
names = dm_inicial.X["Name"]
lista_caracteres = ["(", ")", ",", ".", r"/", r'"']
names_limpios = dm_inicial.eliminar_caracteres(names, lista_caracteres)
df_x_creadas["Name"] = names_limpios

# Edad menor por familia
surname_age = df_x_creadas[["Surname", "Age"]]
edad_menor_surnames = surname_age.groupby("Surname").min()
edad_menor_surnames.columns = ["edad_menor_surnames"]
edad_menor_surnames = df_x_creadas.join(edad_menor_surnames, on="Surname")
edad_menor_surnames = edad_menor_surnames["edad_menor_surnames"]
df_x_creadas["edad_menor_surnames"] = edad_menor_surnames

# Edad mayor por familia
surname_age = df_x_creadas[["Surname", "Age"]]
edad_mayor_surnames = surname_age.groupby("Surname").max()
edad_mayor_surnames.columns = ["edad_mayor_surnames"]
edad_mayor_surnames = df_x_creadas.join(edad_mayor_surnames, on="Surname")
edad_mayor_surnames = edad_mayor_surnames["edad_mayor_surnames"]
df_x_creadas["edad_mayor_surnames"] = edad_mayor_surnames

del df_x_creadas["Age"]

# Cabin
cabin = dm_inicial.X["Cabin"]
df_x_creadas["Cabin_letras"] = dm_inicial.eliminar_digitos(cabin)
df_x_creadas["Cabin_numeros"] = dm_inicial.eliminar_letras(cabin)
df_x_creadas["Cabin_cantidad"] = dm_inicial.contar_palabras(cabin)

# Ticket
ticket = dm_inicial.X["Ticket"]
l_caracteres = ["(", ")", ",", ".", r"/", r'"']
ticket = dm_inicial.eliminar_caracteres(ticket, lista_caracteres=l_caracteres)
df_x_creadas["Ticket_letras"] = dm_inicial.eliminar_digitos(df=ticket)
df_x_creadas["Ticket_numeros"] = dm_inicial.eliminar_letras(df=ticket)
df_x_creadas["Ticket_cantidad"] = dm_inicial.contar_palabras(df=ticket)

# Fare por Ticket
fare = dm_inicial.X["Fare"]
ticket_cantidad = df_x_creadas["Ticket_cantidad"]
df_x_creadas["Fare_dividido_Ticket_cantidad"] = fare / ticket_cantidad

dm_inicial.agregar_y_reemplazar_columnas(df_nuevo=df_x_creadas)
d_filtros = [["porcentaje_nulls", "mayor o igual", 0.3]]
dm_inicial.fillna(valor=-1, d_filtros=d_filtros)
dm_inicial.guardar()

# # GENERO LOS COCIENTES
dm_inicial.generar_cocientes(
    d_filtros=[["dtype", "contains", "float"]],
    inplace=True,
    replace_inf=np.nan,
    replace_nan=-10000,
)

# GENERO LAS PALABRAS BINARIZADAS
dm_inicial.binarizar_palabras(
    umbral=10,
    d_filtros=[["dtype", "contains", "object"]],
    lista_caracteres_a_eliminar=["(", ")", ",", ".", '"'],
    prefix="",
    suffix="",
    inplace=True,
)

# GENERO LOS SMARTBINS SOBRE LAS CONTINUAS
d_filtros = [
    ["dtype", "contains", "float"],
    ["porcentaje_valores_unicos", "mayor", 0.05],
]
dm_inicial.generar_smartbins(
    d_filtros=d_filtros,
    inplace=True,
    parametros_algo=dict(
        n_population=100,
        n_generations=400,
        crossover_proba=1.0,
        mutation_proba=0.5,
        crossover_independent_proba=[0.01, 0.3],
        mutation_independent_proba=[0.01, 0.3],
        tournament_size=5,
        n_random_population_each_gen=30,
        add_mutated_hall_of_fame=True,
        n_gen_no_change=10,
        valor_corte=100000000000.0,
        verbose=True,
    ),
)
dm_inicial.guardar()

# GENERO LAS INTERACCIONES
dm_inicial = DataManager(nombre="titanic", carpeta=carpeta, X=X, y=y)
dm_inicial.cargar()

q = dm_inicial.info["Valor_esperado_ponderado"].quantile(q=0.5)
dm_inicial.generar_interacciones_por_algoritmo_genetico(
    d_filtros=[["Valor_esperado_ponderado", "mayor", q]],
    inplace=True,
    fit_minimo=0.5,
    proba_verdadero=0.2,
    parametros_algo=dict(
        n_population=200,
        n_generations=400,
        crossover_proba=1.0,
        mutation_proba=0.5,
        crossover_independent_proba=[0.01, 0.3],
        mutation_independent_proba=[0.01, 0.3],
        tournament_size=20,
        n_random_population_each_gen=0,
        add_mutated_hall_of_fame=True,
        n_gen_no_change=20,
    ),
)

dm_inicial.guardar()

# CONVIERTO A CATEGORICAS
dm_inicial = DataManager(nombre="titanic", carpeta=carpeta, X=X, y=y)
dm_inicial.cargar()

dm_inicial.convertir_a_binarias(
    d_filtros=[["es_categorica", "igual", True]],
    prefix="Categoricas__",
    suffix="",
    inplace=True,
)

# FILTRO
dm_inicial = DataManager(nombre="titanic", carpeta=carpeta)
dm_inicial.cargar()
dm_inicial.eliminar_features_correlacionados(umbral=1.0, inplace=True)
dm_inicial.guardar()

d_filtros = [["es_numerica", "igual", True]]
dm_final = dm_inicial.generar_nuevo_datamanager(d_filtros=d_filtros, extraer=False)
dm_final.guardar()
