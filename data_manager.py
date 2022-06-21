# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 09:49:19 2022

@author: elope
"""
"""Clase con la cual se parametriza a un analista de machine learning."""

import os
import pandas as pd
import numpy as np
import copy
os.chdir(path=r'C:\Users\elope\OneDrive\Escritorio\disco viejo\lllibs\machine learning')

# GENERALES

from itertools import combinations


# SKLEARN
from sklearn.preprocessing import MinMaxScaler

# LIBRERIA PROPIA
from parsers import Parser
from algoritmo_genetico import AlgoritmoGenetico


# OTROS
import miceforest as mf
from funciones import filtrar_df


# =============================================================================
# FUNCIONES
# =============================================================================
def crear_tabla_frecuencia(serie, y, axis_relativizar=None):
    
    # Calculo la tabla de frequencias
    freq_table = pd.concat([serie.loc[y == etiq].value_counts() for etiq in pd.unique(y) if not np.isnan(etiq)], axis=1).fillna(0)
    freq_table.columns = [etiq for etiq in pd.unique(y) if not np.isnan(etiq)]
    
    # Relativizo para el eje 0
    if axis_relativizar == 0:
        freq_table = freq_table / freq_table.sum(axis=0)
        
    # Relativizo para el eje 1
    elif axis_relativizar == 1:
        freq_table = pd.concat([serie / freq_table.sum(axis=1) for et, serie in freq_table.iteritems()], axis=1)

    return freq_table


def train_test_split_algo_genetico(X, y, X_objetivo, serie_principal, parametros_algo):

    # escalo X
    scaler = MinMaxScaler().fit(X)
    X_analisis_scalado = pd.DataFrame(scaler.transform(X))
    X_analisis_scalado.columns = X.columns

    # escalo el objetivo
    X_objetivo_scalado = pd.DataFrame(scaler.transform(X_objetivo))
    X_objetivo_scalado.columns = X_objetivo.columns
    valores_objetivo = X_objetivo_scalado.corr()

    # Tomo el diccionario
    d_filas = dict()
    d_filas.update({f'filas__{ind}': dict(type=bool) for ind in X_analisis_scalado.index})
   
    def funcion(individuo):
        
        # Tomo los bools del individuo
        lista = list(individuo.values())
        
        # Calculo la media
        X_media = X_analisis_scalado.loc[lista, :].corr()
        ve1 = -abs(valores_objetivo - X_media).mean().mean()
        
        return ve1
    
    # Armo el algoritmo
    algo = AlgoritmoGenetico(dict_variables=d_filas,
                             function=funcion,
                             correr_algortimo=False,
                             **parametros_algo)

    # Corro el algoritmo
    algo.correr_algortimo()

    # Tomo el mejor individuo
    mejor_individuo = algo.mejor_individuo
    
    # Tomo los index objetivos
    index_resultado = list(mejor_individuo.values())
    index_complemento = [not ls for ls in list(mejor_individuo.values())]
    
    if serie_principal == 'train':
        X_train = X.loc[index_resultado]
        y_train = y.loc[index_resultado]
        X_test = X.loc[index_complemento]
        y_test = y.loc[index_complemento] 
        
    elif serie_principal == 'test':
        X_test = X.loc[index_resultado]
        y_test = y.loc[index_resultado]
        X_train = X.loc[index_complemento]
        y_train = y.loc[index_complemento] 
    else:
        raise TypeError('Error en la definicion de la serie_principal.')

    return X_train, X_test, y_train, y_test


# =============================================================================
# Clase
# =============================================================================
class DataManager:

    # =========================================================================
    # Constructor y funciones principales
    # =========================================================================
    def __init__(self, nombre, carpeta, X=None, y=None):
        
        # Storeo los valores
        self.nombre = nombre
        self.carpeta = carpeta
        self.parser = Parser(nombre, carpeta)
        if X is not None:
            self.X = copy.deepcopy(X)

        if y is not None:
            self.y = copy.deepcopy(y)
        
        # Corro el dataset solo en caso de que tenga los dos parametros
        if X is not None and y is not None:
            self.info_dataset()
        
    def agregar_dataset(self, X, y):
        
        # Storeo los parametros
        self.X = copy.deepcopy(X)
        self.y = copy.deepcopy(y)
        self.info_dataset()

    def guardar(self, nombre_archivo=None):
        if not nombre_archivo:
            nombre_archivo = f'{self.nombre}'

        self.parser.pickleo(self.__dict__, nombre_archivo)
        
    def cargar(self, nombre_archivo=None):
        if not nombre_archivo:
            nombre_archivo = f'{self.nombre}'
            
        self.__dict__ = self.parser.unpickleo(nombre_archivo)

        
# =========================================================================
# Funciones para obtener info del dataset
# =========================================================================
    def es_dummy(self, serie):
        
        # Si tiene solo dos valores unicos es dummy
        res = len(pd.unique(serie)) == 2
        return res

    def es_categorica(self, serie, umbral=0.1):
    
        if self.es_dummy(serie):
            res = False
            
        elif serie.dtype.name == 'category':
            res = True
        
        else: 
        
            if type(umbral) is int:
                res =  umbral >= len(pd.unique(serie)) 
            
            elif type(umbral) is float:
                res = (len(pd.unique(serie)) / serie.shape[0]) < umbral
            
            else:
                raise TypeError('Error en umbral. Debe ser int mayor a 0 o float entre 0. y 1.')

        return res

    def es_continua(self, serie, umbral=0.9):
        
        # Contrasto
        cond1 = (not self.es_categorica(serie, umbral=1.-umbral) and not self.es_dummy(serie))
        cond2 = ('float' in serie.dtype.name)
        res = cond1 and cond2
        return res
    
    def info_serie(self, serie, y):
        
        # Creo el objeto
        d = dict()
        
        # Incorporo los datos
        d['name'] = serie.name
        d['dtype'] = serie.dtype.name
        d['cantidad_nulls'] = sum(serie.isnull())
        d['cantidad_not_nulls'] = serie.shape[0] - sum(serie.isnull())
        d['tiene_nulls'] = serie.isnull().any()
        d['cant_observaciones'] = serie.shape[0]
        d['porcentaje_nulls'] = d['cantidad_nulls'] / d['cant_observaciones']
        d['cant_valores_unicos'] = len(pd.unique(serie))
        d['porcentaje_valores_unicos'] = d['cant_valores_unicos'] / d['cant_observaciones'] 
        d['Valor_esperado_ponderado'] = self.calcular_valor_esperado_ponderado(serie, y)
        d['es_dummy'] = self.es_dummy(serie)
        d['es_categorica'] = self.es_categorica(serie)
        d['es_continua'] = self.es_continua(serie)
        d['es_numerica'] = 'float' in d['dtype'] or 'int' in d['dtype']
        d['es_object'] = 'object' in d['dtype']
        d['es_category'] = 'category' in d['dtype']

        # Solo las numericas
        if d['es_numerica']:

            d['min'] = min(serie)
            d['media'] = serie.mean()
            d['max'] = max(serie)
            d['mediana'] = serie.median()
            d['moda'] = serie.mode()[0]
            d['correlacion_con_y'] = pd.DataFrame(serie).corrwith(other=y)[0]
            d['todas_strings'] = False
            
        # Las no numericas me fijo si son todas strings
        elif d['es_object']:
            d['todas_strings'] = pd.DataFrame(serie).applymap(type).eq(str).all()[0]

        return pd.DataFrame([d])

    def calcular_frecuencias(self, serie, y):
    
        # Creo el diccionario
        d = dict()
        
        # Calculo las frecuencias
        d['freq_absoluta'] = crear_tabla_frecuencia(serie, y, axis_relativizar=None)  
        d['freq_relativa_0'] = crear_tabla_frecuencia(serie, y, axis_relativizar=0)            
        d['freq_relativa_1'] = crear_tabla_frecuencia(serie, y, axis_relativizar=1)
        
        # Concateno en una sola tabla
        d['freq'] = pd.concat([d['freq_absoluta'], d['freq_relativa_0'], d['freq_relativa_1']], axis=1)
    
        return d
        
    def info_dataset(self):
        
        # Calculo el info
        self.info = pd.concat([self.info_serie(serie=serie, y=self.y) for nombre, serie in self.X.iteritems()]).set_index('name', drop=False)
        
        # Calculo las frecuencias
        self.frecuencias = {nombre: self.calcular_frecuencias(serie, self.y) for nombre, serie in self.X.iteritems()}
        
        # Calculo la matriz de correlacion
        self.corr = self.X.corr()

    def consultar(self, d_filtros=None):
    
        # Filtro las columnas
        if d_filtros:
            d = {f'{a[0]}': [a[2], a[1]]  for a in d_filtros}
            columnas = list(filtrar_df(df=self.info, diccionario=d).index)
        else:
            columnas = self.X.columns
        
        # Filtro el dataset
        dataset = self.X[columnas]
            
        return [columnas, dataset]
    
    def consultar_columnas(self, d_filtros=None):
        [columnas, dataset] = self.consultar(d_filtros)
        return columnas
    
    def consultar_dataset(self, d_filtros=None):
        [columnas, dataset] = self.consultar(d_filtros)
        return dataset

# %%=======================================================================
# GESTION DE DMs
# =========================================================================
    def eliminar_features_por_lista(self, columnas):
        for col in columnas:
            del self.X[col]

        self.info_dataset()
        
    def generar_nuevo_datamanager(self, d_filtros, extraer=False):
        
        # Consulto el dataset
        [columnas, dataset] = self.consultar(d_filtros)
        
        # Creo el nuevo dataset
        otro_dm = DataManager(f'{self.nombre}', self.carpeta, dataset, self.y)
        
        # Elimino en el original si corresponde
        if extraer:
            self.eliminar_features_por_lista(columnas=columnas)
        
        return otro_dm
    
    def separar_por_columna_info(self, nombre_columna_info):

        b = {a: self.generar_nuevo_datamanager(d_filtros=[[nombre_columna_info, 'igual', a]], extraer=False)
             for a in list(pd.unique(self.info[nombre_columna_info]))}
    
        return b

    def copiar_features_a_otro_datamanager(self, d_filtros=None, otro_dm=None, eliminar_en_origen=False):
        
        [columnas, dataset] = self.consultar(d_filtros)
        
        if not otro_dm:
            otro_dm = DataManager(f'{self.nombre}', carpeta=self.carpeta, X=dataset, y=self.y)
        else:
            otro_dm.agregar_y_reemplazar_columnas(df_nuevo=dataset)
        
        if eliminar_en_origen: self.eliminar_features_por_lista(columnas=columnas)
        
        return otro_dm

    def agregar_y_reemplazar_columnas(self, df_nuevo, prefix='', suffix=''):
        
        if type(df_nuevo) is pd.core.series.Series:
            df_nuevo = pd.DataFrame(df_nuevo)
            
        if not hasattr(self, 'X'):
            self.X = pd.DataFrame()

        if not hasattr(self, 'y'):
            self.X = pd.Series()

        df_nuevo.columns = [f'{prefix}{col}{suffix}' for col in df_nuevo.columns]

        for col in df_nuevo.columns:
            self.X[col] = df_nuevo[col]
        
        self.info_dataset()

    def agregar_otros_datamanager(self, otros, d_filtros=None):
        
        if type(otros) is list:
            lista_otros = otros
        else:
            lista_otros = [otros]

        for otro in lista_otros:
            x = otro.consultar_dataset(d_filtros=d_filtros)
            self.agregar_y_reemplazar_columnas(df_nuevo=x)

# =========================================================================
# Gestion de columnas
# =========================================================================
    def fillna(self,
               valor,
               d_filtros=None,
               prefix='',
               suffix='',
               cantidad_datasets=5,
               inplace=True,
               verbose=False):

        # Tomo el dataset
        X = self.consultar_dataset(d_filtros)      

        # para el tipo mice
        if valor == 'mice':
    
            if X.isnull().any().any():
    
                # Convierto a categoricas las object
                etiqs = dict()
                for i, serie in X.iteritems():
                    if serie.dtype.name == 'object':
                        factorize2 = pd.factorize(serie)
                        X[i] = pd.Series(factorize2[0], index=X.index)
                        etiqs[i] = factorize2[1]
                
                X = X.replace(-1, np.nan)
                    
                # Creo el kernel
                kernel = mf.ImputationKernel(data=X,
                                             save_all_iterations=True,
                                             random_state=0)
                
                # Corro el algoritmo mice
                kernel.mice(cantidad_datasets, verbose=verbose)
                
                # Completo el dataset con los nans
                variables_completadas = kernel.complete_data(cantidad_datasets-1)
                
                # Devuelvo a object las object convertidas
                for (i, variable_completada), (j, variable_original) in zip(variables_completadas.iteritems(), X.iteritems()):
                    if i in list(etiqs.keys()): 
                        d = {ind: etiqueta for ind, etiqueta in enumerate(etiqs[i])}
                        variables_completadas[i] = variable_completada.replace(d)
        
        # Para los demás tipos completo con fillna segun el tipo de formato que sea
        else:
            
            l = list()
            for a, serie in X.iteritems():
            
                if pd.DataFrame(serie).applymap(type).eq(str).all()[0]:
                    variable_completada = serie.fillna(str(valor))
                    
                elif pd.DataFrame(serie).applymap(type).eq(int).all()[0]:
                    variable_completada = serie.fillna(int(valor))
                    
                else:
                    variable_completada = serie.fillna(float(valor))
                
                l.append(variable_completada)
            
            variables_completadas = pd.concat(l, axis=1)

        if inplace:
            self.agregar_y_reemplazar_columnas(df_nuevo=variables_completadas, prefix=prefix, suffix=suffix)
        else:
            # Creo el dm nuevo con los resultados
            self_object = DataManager(f'{prefix}{self.nombre}{suffix}', self.carpeta, variables_completadas, self.y)
            return self_object

    def convertir_a_categoricas(self,
                                d_filtros=None,
                                prefix='',
                                suffix='',
                                inplace=True,
                                conservar_nan=True):
        
        # Tomo las columnas que voy a utilizar
        df_categoricas = self.consultar_dataset(d_filtros)     
    
        # categorizo y concateno
        series_categoricas = list()
        for a in df_categoricas.columns:
            cat = pd.factorize(df_categoricas[a])
            cat_df = pd.DataFrame(cat[0])
            cat_df.columns = [f'{prefix}{a}{suffix}']
            series_categoricas.append(cat_df)
    
        res = pd.concat(series_categoricas, axis=1)
        res.set_index(df_categoricas.index, inplace=True)
        
        if conservar_nan: res = res.replace(-1, np.nan)
        
        if inplace:
            self.agregar_y_reemplazar_columnas(df_nuevo=res, prefix=prefix, suffix=suffix)
        else:
            # Creo el dm nuevo con los resultados
            self_object = DataManager(f'{prefix}{self.nombre}{suffix}', self.carpeta, res, self.y)
            return self_object

    def convertir_a_binarias(self,
                             d_filtros=None,
                             prefix='',
                             suffix='',
                             inplace=True):
        
        # Calculo la lista de columnas
        df_binarias = self.consultar_dataset(d_filtros)     
    
        # Binarizo y concateno
        series_binarias = list()
        for a in df_binarias.columns:
            series_binarizadas = pd.get_dummies(df_binarias[a], dummy_na=False, drop_first=False)
            series_binarizadas.columns = [f'{prefix}{a}_{col}{suffix}' for col in series_binarizadas.columns]
            series_binarias.append(series_binarizadas)
    
        res = pd.concat(series_binarias, axis=1)
        res.set_index(df_binarias.index, inplace=True)
    
        if inplace:
            self.agregar_y_reemplazar_columnas(df_nuevo=res, prefix=prefix, suffix=suffix)
        else:
            # Creo el dm nuevo con los resultados
            self_object = DataManager(f'{prefix}{self.nombre}{suffix}', self.carpeta, res, self.y)
            return self_object
    
    def eliminar_features_correlacionados(self,
                                          umbral,
                                          d_filtros=None,
                                          prefix='',
                                          suffix='',
                                          inplace=True):
        
        # Obtengo el dataset sobre el que voy a trabajar
        X = self.consultar_dataset(d_filtros)
        
        # Creo la matriz de correlacion
        corr_matrix = X.corr().abs()
    
        # Selecciono el triángulo superior de la matriz de correlacion
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
        # Busco el index de las columnas de features con correlacion mayor al umbral
        to_drop = [column for column in upper.columns if any(upper[column] > umbral)]
    
        # Elimino los features
        res = X.drop(X[to_drop], axis=1)
    
        if inplace:
            self.agregar_y_reemplazar_columnas(df_nuevo=res, prefix=prefix, suffix=suffix)
        else:
            # Creo el dm nuevo con los resultados
            self_object = DataManager(f'{prefix}{self.nombre}{suffix}', self.carpeta, res, self.y)
            return self_object

    def calcular_interacciones(self, df, cantidad_combinaciones, inplace=True):
        
        df = combinar_series(df, cantidad_combinaciones)
    
        if inplace:
            self.agregar_y_reemplazar_columnas(df_nuevo=df, prefix='Interacciones__')
        else:
            # Creo el dm nuevo con los resultados
            self_object = DataManager(f'Interacciones__{self.nombre}', self.carpeta, df, self.y)
            return self_object
        
    def binarizar_palabras(self,
                           umbral,
                           d_filtros=None,
                           inplace=True,
                           lista_caracteres_a_eliminar=None,
                           prefix='',
                           suffix=''): 
        
        # Obtengo el dataset sobre el que voy a trabajar
        df = self.consultar_dataset(d_filtros)    
        
        # Creo el df resultado
        df_resultado = pd.DataFrame()
        
        # Transformo en df
        if type(df) is pd.core.series.Series:
            df = pd.DataFrame(df)
            
        if lista_caracteres_a_eliminar:
            df = self.eliminar_caracteres(X=df, lista_caracteres=lista_caracteres_a_eliminar, regex=False)
            
        # Para cada columna
        for a, col in df.iteritems():
           
            # Calculo
            a = col.str.get_dummies(sep=' ')
            s = a.sum().sort_values(ascending=False)
            
            # Tomo los que entran en el umbral
            if umbral > 0:
                
                if type(umbral) is int:
                    s3 = s.head(umbral)
                    
                elif type(umbral) is float and umbral >= 0. and umbral <= 1.:
                    cantidad = s.shape[0]
                    umbral2 = int(cantidad * umbral)
                    s3 = s.head(umbral2)
                
                else:
                    raise TypeError ('Error en la definicion de umbral.')
            
                for key, cantidad in s3.iteritems():
                    df_resultado[f'Palabra_binarizada_ {col.name}_{key}'] = a[key].astype(float)
    
        if inplace:
            self.agregar_y_reemplazar_columnas(df_nuevo=df_resultado, prefix=prefix, suffix=suffix)
        else:
            # Creo el dm nuevo con los resultados
            self_object = DataManager(f'{prefix}{self.nombre}{suffix}', self.carpeta, df_resultado, self.y)
            return self_object

    def generar_cocientes(self,
                           d_filtros=None,
                           inplace=True,
                           replace_inf=np.nan,
                           replace_nan=-10000,
                           prefix='',
                           suffix=''):
        
        # Obtengo el dataset sobre el que voy a trabajar
        df = self.consultar_dataset(d_filtros)   
        
        lista_df = [s for i, s in df.iteritems()]
        
        # calculo las combinaciones
        cantidad_combinaciones = 2
        combinaciones = list(combinations(list(range(len(lista_df))), min(cantidad_combinaciones, len(lista_df))))
            
        cocientes = pd.concat([df.iloc[:, comb[0]] / df.iloc[:, comb[1]] for comb in combinaciones], axis=1)
        nombres = [f'{df.iloc[:, comb[0]].name}_dividido_{df.iloc[:, comb[1]].name}' for comb in combinaciones]
        cocientes.columns = nombres
    
        cocientes = cocientes.replace(np.inf, replace_inf)
        cocientes = cocientes.replace(np.nan, replace_nan)      
    
        if inplace:
            self.agregar_y_reemplazar_columnas(df_nuevo=cocientes, prefix=prefix, suffix=suffix)
        else:
            # Creo el dm nuevo con los resultados
            self_object = DataManager(f'{prefix}{self.nombre}{suffix}', self.carpeta, cocientes, self.y)
            return self_object
    
# =============================================================================
# Strings
# =============================================================================
    def eliminar_caracteres(self, X=None, lista_caracteres=['(', ')', ',', '.'], regex=False):
        
        # Creo el df resultado
        df_resultado = copy.deepcopy(X)
                
        # Transformo en df
        if type(df_resultado) is pd.core.series.Series:
            df_resultado = pd.DataFrame(df_resultado)
            
        
        for a, col in df_resultado.iteritems():
            for caracter in lista_caracteres:             
                col = col.str.replace(caracter, '', regex=regex)
                df_resultado[col.name] = col
    
        return df_resultado
    
    def eliminar_digitos(self, df):
        return df.str.replace('\d+', '', regex=True)
    
    def eliminar_letras(self, df):    
        return df.str.replace('[A-Z]', '', regex=True)
    
    def contar_palabras(self, df):
        return df.str.split(expand=True).fillna(value=np.nan).T.count()
    
    def contar_repeticiones(self, df):

        if df.shape[1] > 1: raise TypeError('El df debe tener solo una columna')
        
        df2 = pd.Series(np.ones(df.shape[0]))
        df3 = pd.concat([df, df2], axis=1)
        df4 = df3.groupby(list(df.columns)).count()[0]
        df4.name = df.columns[0]
        df5 = df.set_index(df.columns[0]).join(df4)
        df5.columns = [f'Count_{df4.name}']
        df5.set_index(df.index, inplace=True)
        
        return df5
    
    def calcular_cocientes(self, 
                           df,
                           replace_inf=np.nan,
                           replace_nan=-10000):
            
        
        lista_df = [s for i, s in df.iteritems()]
        
        # calculo las combinaciones
        cantidad_combinaciones = 2
        combinaciones = list(combinations(list(range(len(lista_df))), min(cantidad_combinaciones, len(lista_df))))
            
        cocientes = pd.concat([df.iloc[:, comb[0]] / df.iloc[:, comb[1]] for comb in combinaciones], axis=1)
        nombres = [f'{df.iloc[:, comb[0]].name}_dividido_{df.iloc[:, comb[1]].name}' for comb in combinaciones]
        cocientes.columns = nombres
    
        cocientes = cocientes.replace(np.inf, replace_inf)
        cocientes = cocientes.replace(np.nan, replace_nan)      
        
        return cocientes

# %%=========================================================================
# Metricas
# =========================================================================
    def calcular_gini(self, df, y):
        
        bins_1 = df.loc[pd.notnull(y)].value_counts()
        bins_2 = df.loc[y==1.].value_counts().reindex(bins_1.index).fillna(0.)
        bins_3 = bins_2 / bins_1
        
        x = (bins_3 / bins_3.sum()).values
       
        # Calculo gini
        mad = np.abs(np.subtract.outer(x, x)).mean()
        
        # Relative mean absolute difference
        rmad = mad/np.mean(x)
        
        # Gini coefficient
        H = 0.5 * rmad
   
        return H
    
    def calcular_gini_sobre_cantidad_bins(self, df, y):
        
        bins_1 = df.loc[pd.notnull(y)].value_counts()
        bins_2 = df.loc[y==1.].value_counts().reindex(bins_1.index).fillna(0.)
        bins_3 = bins_2 / bins_1
        
        x = (bins_3 / bins_3.sum()).values
       
        # Calculo gini
        mad = np.abs(np.subtract.outer(x, x)).mean()
        
        # Relative mean absolute difference
        rmad = mad/np.mean(x)
        
        # Gini coefficient
        H = 0.5 * rmad
   
        return H / len(x)
    
    def calcular_gini_ponderado(self, df, y):
        
        x_0 = crear_tabla_frecuencia(df, y, axis_relativizar=0.)[1.]
        x_1 = crear_tabla_frecuencia(df, y, axis_relativizar=1.)[1.]
        
        x3  = x_0 * x_1
        x = (x3 / x3.sum()).values
       
        # Calculo gini
        mad = np.abs(np.subtract.outer(x, x)).mean()
        
        # Relative mean absolute difference
        rmad = mad/np.mean(x)
        
        # Gini coefficient
        H = 0.5 * rmad
   
        return H / len(x)
    
    def indice_Hirschman_Herfindhal(self, df3):
        return df3.pow(2).sum()

    def indice_Hirschman_Herfindhal_normalizado(self, df3):
        H = self.indice_Hirschman_Herfindhal(df3)
        n = df3.shape[0]        
        return (H - (1./n)) / (1 - (1./n))
    
    def calcular_indice_concentracion(self, df, y, normalizar=True):

        bins_1 = df.loc[pd.notnull(y)].value_counts()
        bins_2 = df.loc[y==1.].value_counts().reindex(bins_1.index).fillna(0.)
        bins_3 = bins_2 / bins_1
        
        x = bins_3 / bins_3.sum()
        
        if len(x) == 1: H = 0
        else:
            if normalizar:
                H =  self.indice_Hirschman_Herfindhal_normalizado(x)
            else:
                H =  self.indice_Hirschman_Herfindhal(x)
                    
        if H is np.nan: return 0.
        else: return H
        
    def calcular_indice_concentracion_ponderado(self, df, y, normalizar=True):

        x_0 = crear_tabla_frecuencia(df, y, axis_relativizar=0.)[1.]
        x_1 = crear_tabla_frecuencia(df, y, axis_relativizar=1.)[1.]
        
        x  = x_0 * x_1
        
        if len(x) == 1: H = 0
        else:
            if normalizar:
                H =  self.indice_Hirschman_Herfindhal_normalizado(x)
            else:
                H =  self.indice_Hirschman_Herfindhal(x)
                    
        if H is np.nan: return 0.
        else: return H
        
    def calcular_valor_esperado_ponderado(self, df, y, penalizar_valores_unicos=True):

        x_None = crear_tabla_frecuencia(df, y, axis_relativizar=None)
        x_0 = x_None / x_None.sum(axis=0)
        x_1 = crear_tabla_frecuencia(df, y, axis_relativizar=1.)

        freq_total = x_None.sum()
        freq_rel = freq_total / freq_total.sum()
        
        x  = x_0 * x_1
        x2 = x.sum()
        H = (x2 * freq_rel).sum()
        
        if penalizar_valores_unicos:
            H = H * (1 - x_0.shape[0] / freq_total.sum())
                    
        return H
        
# %%=============================================================================
# Metodos para crear nuevos DM
# =============================================================================
    def calcular_bins_inteligentes(self,
                                   df,
                                   y,
                                   rango_minimo=0.15,
                                   rango_maximo=0.3,
                                   freq_minima = 0.2,
                                   freq_maxima = 0.5,
                                   cant_maxima_bins=10,
                                   parametros_algo=dict(n_population=100,
                                                        n_generations=200,
                                                        crossover_proba=1.,
                                                        mutation_proba=0.5,
                                                        crossover_independent_proba=0.2,
                                                        mutation_independent_proba=0.1,
                                                        tournament_size=10,
                                                        n_random_population_each_gen=20,
                                                        add_mutated_hall_of_fame=True,
                                                        n_gen_no_change=10,
                                                        verbose=True)):
        
        
        if type(df) is not pd.core.series.Series: raise TypeError('df debe ser un pandas series')
        
        # Ordeno el df y tomo el n
        df_ordenado = df.sort_values(ascending=True)
        y_ordenado = y.loc[df_ordenado.index]
        df_ordenado = df_ordenado.reset_index(drop=True)
        y_ordenado = y_ordenado.reset_index(drop=True)

        n = df_ordenado.shape[0]
                
        def individuo_a_cortes(df_ordenado, individuo):

            # Creo la lista de resultados
            cortes2 = list()
            indices2 = list()
            
            # Tomo el valor inicial y final y creo el rango
            indice_inicial = 0
            valor_inicial = df_ordenado.iloc[indice_inicial]
            
            indice_ultimo = n - 1
            valor_final = df_ordenado.iloc[indice_ultimo]
            
            rango = valor_final - valor_inicial
            
            # Agrego el primer valor
            cortes2.extend([valor_inicial])
            indices2.extend([indice_inicial])
      
            # Tomo los elementos del individuo
            d_segmentos = {k: v for k, v in individuo.items() if 'Frecuencia__' in k}
            
            # Genero los valores iniciales
            valor_ultimo = valor_inicial
            indice_def = indice_inicial
            
            for k, freq in d_segmentos.items(): # k, freq = next(d_segmentos.items())
            
                # Calculo el valor último
                valor_ultimo = valor_ultimo + int(freq * rango)

                if valor_ultimo >= valor_final: break
                
                else: 
                
                    # Corrijo por frecuencia (topeo)
                    indice_final = df_ordenado.loc[df_ordenado>=valor_ultimo].index[0]
                    indice_minimo = indice_def + int(n *  freq_minima)
                    indice_maximo = indice_def + int(n *  freq_maxima)
                    
                    if indice_final >= indice_maximo:
                        indice_def = min(indice_maximo, indice_final)
                        
                        if indice_def > indice_ultimo: break
                    
                        valor_ultimo = df_ordenado.loc[indice_def]
                        
                    elif indice_final <= indice_minimo:
                        indice_def = max(indice_minimo, indice_final)
                        
                        if indice_def > indice_ultimo: break
                                            
                        valor_ultimo = df_ordenado.loc[indice_def] 
                    else:
                        indice_def = indice_final
                
                    if ((indice_ultimo - indice_def) / indice_ultimo) < freq_minima: break

                    # Storeo en la lista
                    cortes2.extend([valor_ultimo])
                    indices2.extend([indice_def])
            
            # agrego el final
            cortes2.extend([valor_final])
            indices2.extend([indice_ultimo])
                         
            return cortes2

        def func(individuo):
        
            # Calculo los cortes
            cortes2 = individuo_a_cortes(df_ordenado, individuo)
            
            # Calculo los bins
            if len(cortes2) == 2:
                H = 0.
            else:
                bins = pd.cut(df, cortes2, duplicates='drop', include_lowest='True')
                # H = self.calcular_gini(bins, y)
                H = self.calcular_valor_esperado_ponderado(bins, y, penalizar_valores_unicos=True)
                
                
            return H 

        # Armo el diccionario
        d_total = dict()
        d_total.update({f'Frecuencia__{col}': dict(type=float,
                                                   low=rango_minimo,
                                                   high=rango_maximo)
                        for col in range(cant_maxima_bins)})
       
        # Creo el algoritmo y lo corro
        algo = AlgoritmoGenetico(dict_variables=d_total,
                                 function=func,
                                 correr_algortimo=False,
                                 **parametros_algo)
        
        algo.correr_algortimo()
        
        
        # Check
        individuo = algo.mejor_individuo
           
        # Calculo los cortes
        cortes2 = individuo_a_cortes(df_ordenado, individuo)
        
        # Calculo los bins
        bins = pd.cut(df, cortes2, duplicates='drop', include_lowest='True').squeeze()
        
        return bins
    
    def generar_smartbins(self,
                          d_filtros=None,
                          inplace=True,
                          rango_minimo = 0.,
                          rango_maximo = 1.,
                          freq_minima = 0.2,
                          freq_maxima = 1.,
                          cant_maxima_bins=int(1/.15),                                    
                          parametros_algo=dict(n_population=100,
                                               n_generations=400,
                                               crossover_proba=[0.5, 1.],
                                               mutation_proba=[0.5, 1.],
                                               crossover_independent_proba=[0.05, 0.3],
                                               mutation_independent_proba=[0.05, 0.3],
                                               tournament_size=4,
                                               n_random_population_each_gen=50,
                                               add_mutated_hall_of_fame=True,
                                               n_gen_no_change=10,
                                               valor_corte=100000000000.,
                                               verbose=True)):
                                                 
        # Obtengo el dataset sobre el que voy a trabajar
        df_temp = self.consultar_dataset(d_filtros)
        
        l_total = list()
        for nombre, df in df_temp.iteritems(): # nombre, df = next(df_temp.iteritems())
            print(nombre)
            bins = self.calcular_bins_inteligentes(df,
                                                   self.y,
                                                   rango_minimo,
                                                   rango_maximo,
                                                   freq_minima,
                                                   freq_maxima,
                                                   cant_maxima_bins,                                    
                                                   parametros_algo)
        
            l_total.append(bins)
    
        df_total = pd.concat(l_total, axis=1)

        if inplace:
            self.agregar_y_reemplazar_columnas(df_nuevo=df_total, prefix='Smartbins__')
        else:
            # Creo el dm nuevo con los resultados
            self_object = DataManager(f'Smartbins__{self.nombre}', self.carpeta, df_total, self.y)
            return self_object
        
    def generar_interacciones_por_algoritmo_genetico(self,
                                                     d_filtros=None,
                                                     inplace=True,
                                                     fit_minimo=0.5,
                                                     proba_verdadero=0.2,
                                                     parametros_algo=dict(n_population=200,
                                                                          n_generations=200,
                                                                          crossover_proba=1.,
                                                                          mutation_proba=0.75,
                                                                          crossover_independent_proba=[0.01, 0.15],
                                                                          mutation_independent_proba=[0.01, 0.15],
                                                                          tournament_size=20,
                                                                          n_random_population_each_gen=50,
                                                                          add_mutated_hall_of_fame=True,
                                                                          n_gen_no_change=3)):
   
        dataset = self.consultar_dataset(d_filtros)
                
        if not proba_verdadero:
            proba_verdadero=0.5

        p = [proba_verdadero, 1.-proba_verdadero]
        
        # Armo el diccionario de las columnas
        d_columnas = dict()
        d_columnas.update({f'columnas__numeroColumna_{col}': dict(type=list, lista=[True, False], p=p) 
                           for col in range(int(dataset.shape[1]))})

        # Armo la funcion objetivo
        def func(individuo):
        
            if sum(individuo.values()) == 0:
                H = 0.

            else:            
                # Tomo las columnas que voy a combinar
                df = dataset.loc[:, list(individuo.values())]
                cantidad_combinaciones = df.shape[1]
                
                # Combino
                df_combinado = combinar_series(df, cantidad_combinaciones)
            
                # Calculo la metrica
                H = self.calcular_valor_esperado_ponderado(df_combinado, self.y)
                
            return H 

        # Creo el algoritmo y lo corro
        algo = AlgoritmoGenetico(dict_variables=d_columnas,
                                 function=func,
                                 correr_algortimo=False,
                                 **parametros_algo)

        algo.correr_algortimo()

        # Tomo los valores del salon de la fama como resultado
        l_inds = list()
        l_fits = list()
        for individuo in algo.poblacion_salon_de_la_fama: # individuo = algo.poblacion_salon_de_la_fama[0]
            
            df = dataset.loc[:, list(individuo.values())]
            cantidad_combinaciones = df.shape[1]
        
            if cantidad_combinaciones == 0: continue
        
            df_combinado = combinar_series(df=df, cantidad_combinaciones=cantidad_combinaciones)
            l_inds.append(df_combinado)
            l_fits.append(self.calcular_valor_esperado_ponderado(df_combinado, self.y))
    
        # Uno los df_combinados
        df_total = pd.concat(l_inds, axis=1)
        
        # Storeo o retorno segun corresponda
        if inplace:
            self.agregar_y_reemplazar_columnas(df_nuevo=df_total, prefix='Interacciones__')
        else:
            # Creo el dm nuevo con los resultados
            self_object = DataManager(f'Interacciones__{self.nombre}', self.carpeta, df_total, self.y)
            return self_object


# =============================================================================
# FUNCIONES COMPLEMENTARIAS
# =============================================================================
def combinar_series(df, cantidad_combinaciones):
    
    # Valido los index
    lista_df = [s for i, s in df.iteritems()]
        
    # calculo las combinaciones
    comb = list(combinations(list(range(len(lista_df))), min(cantidad_combinaciones, len(lista_df))))
    
    # Armo las listas
    names = list()
    values = list()
    for i, tup in enumerate(comb):
        names_2 = list()
        values_2 = list()
        for el in tup:
            names_2.append(lista_df[el].name)
            values_2.append(lista_df[el].values)
        names.append(names_2)
        values.append(values_2)
            
    # Combino
    series = list()
    for c, n in zip(values, names):
        nueva_serie = pd.Series([f'{elements}' for elements in zip(*c)], index=lista_df[0].index)
        nueva_serie.name = f'{n}'
        series.append(nueva_serie)
    
    return pd.concat(series, axis=1)
