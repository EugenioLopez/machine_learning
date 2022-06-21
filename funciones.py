# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 08:47:17 2022

@author: elope
"""
from datetime import datetime
import copy
import pandas as pd

# =============================================================================
# FUNCIONES PARA EL MANEJO DE DATAFRAMES
# =============================================================================
def filtrar_columnas_por_patron(
    df,
    patron,
    es_interseccion=False,
    patron_exclusion=None,
    es_interseccion_exclusion=False,
):

    if type(patron) is str:  # Corrijo el patron si es string
        p = list()
        p.append(patron)
        patron = p
    if es_interseccion:  # Filtro por cada patron
        b = copy.copy(df)
        for p in patron:
            b = b.filter(regex=p, axis=1)
    else:

        a = list()
        for p in patron:
            a.append(df.filter(regex=p, axis=1))
        b = pd.concat(a, axis=1)  # Uno las columnas obtenidas
    c = b.loc[:, ~b.columns.duplicated()]  # Elimino las columnas repetidas

    if patron_exclusion:  # Filtro las excluidas

        c2 = filtrar_columnas_por_patron(
            c,
            patron=patron_exclusion,
            es_interseccion=es_interseccion_exclusion,
            patron_exclusion=None,
        )
        c = c.drop(c2.columns, axis=1)
    return c


def filtrar_df(df, diccionario):

    v_total = copy.deepcopy(df)

    for k, v in diccionario.items():

        if type(v) is list:

            v0 = v[0]
            v1 = v[1]

            if type(v0) is str:

                if v1 == "contains":
                    v_total = v_total.loc[v_total[k].str.contains(v0, case=False)]
                elif v1 == "igual":
                    v_total = v_total.loc[v_total[k] == v0]
                else:
                    raise ValueError(
                        "Error en el tipo de funcion para el tipo de dato."
                    )
            else:

                if v1 == "igual":
                    v_total = v_total.loc[v_total[k] == v0]
                elif v1 == "distinto":
                    v_total = v_total.loc[v_total[k] != v0]
                elif v1 == "mayor":
                    v_total = v_total.loc[v_total[k] > v0]
                elif v1 == "mayor o igual":
                    v_total = v_total.loc[v_total[k] >= v0]
                elif v1 == "menor":
                    v_total = v_total.loc[v_total[k] < v0]
                elif v1 == "menor o igual":
                    v_total = v_total.loc[v_total[k] <= v0]
                else:
                    raise ValueError(
                        "Error en el tipo de funcion para el tipo de dato."
                    )
        else:

            if type(v) is str:
                v_total = v_total.loc[v_total[k].str.contains(v, case=False)]
            else:
                v_total = v_total.loc[v_total[k] == v]
    return v_total


def cruzar_dos_tablas(df1, df2, on, patron, how="left", rsuffix="old"):

    # Copio la df1 base
    df1_copia = copy.deepcopy(df1)
    df1_copia.set_index(on, inplace=True)

    # Copio la df2 que voy a cruzar
    df2_copia = copy.deepcopy(df2)
    df2_copia.set_index(on, inplace=True)

    # Filtro las columnas que voy a usar
    if patron:
        df2_copia = filtrar_columnas_por_patron(
            df2_copia, patron=patron, es_interseccion=False
        )
    # Cruzo
    df3 = df1_copia.join(df2_copia, on=on, how=how, rsuffix=rsuffix)
    df4 = df3.loc[:, ~df3.columns.duplicated()]
    df4.reset_index(inplace=True, drop=False)

    return df4


def date2str(t):
    """
    Función que convierte una fecha en un string con determinado formato.

    Parameters
    ----------
    t : TYPE. datetime
        DESCRIPTION. fecha tipo datetime

    Returns
    -------
    String resultado.

    """
    if t.year < 10:
        year = f"0{t.year}"
    else:
        year = t.year
    if t.month < 10:
        month = f"0{t.month}"
    else:
        month = t.month
    if t.day < 10:
        day = f"0{t.day}"
    else:
        day = t.day
    if t.hour < 10:
        hour = f"0{t.hour}"
    else:
        hour = t.hour
    if t.minute < 10:
        minute = f"0{t.minute}"
    else:
        minute = t.minute
    if t.second < 10:
        second = f"0{t.second}"
    else:
        second = t.second
    if t.microsecond < 10:
        second = f"0{t.microsecond}"
    else:
        microsecond = t.microsecond
    string = f"{year}_{month}_{day}_{hour}_{minute}_{second}_{microsecond}"
    return string


def str2date(string):
    """
    Funcion que convierte un string en un datetime.

    Parameters
    ----------
    string : TYPE str
        DESCRIPTION. String

    Returns
    -------
    None. Datetime

    """
    lista = str.split(string, sep="_")
    lista2 = [int(a) for a in lista]
    return datetime(*lista2)
