# -*- coding: utf-8 -*-
# %%
"""Clase con la cual se parametriza a un analista de machine learning."""
import os
import sys
import pandas as pd
import numpy as np
import copy
from datetime import datetime

from parsers import Parser

# from sklearn.preprocessing import LabelBinarizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

# from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

# from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer

# from sklearn.metrics import roc_curve
# from sklearn.metrics import classification_report
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import matthews_corrcoef
# from sklearn.metrics import log_loss
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import silhouette_score
# from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler

# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

# from graficos import heatmap
# from graficos import graficar_tabla
# from graficos import grafico_con_slide

from ingenieria_modelos import generar_estimador
from ingenieria_modelos import generar_parametros_algoritmo_genetico
from algoritmo_genetico import AlgoritmoGenetico

from lightgbm import LGBMClassifier
# import miceforest as mf

# from funciones import filtrar_columnas_por_patron
from funciones import date2str

# from funciones import str2date
from funciones import filtrar_df

from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

from sklearn.metrics import homogeneity_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import completeness_score
from sklearn.cluster import KMeans

import plotly.io as pio
import plotly.express as px
pio.renderers.default = "browser"

# %%=============================================================================
# CLASE
# =============================================================================
class Analista:
    """
    Clase que simula a un analista de machine learning.
    """

    # =========================================================================
    # CONSTRUCTOR
    # =========================================================================
    def __init__(self, nombre, carpeta):

        # Copio los valores

        self.nombre = nombre
        self.carpeta = carpeta
        self.parser = Parser(nombre, carpeta)

        # Agrego atributos adicionales que voy a utilizar después
        self.nueva_version = None
        self.nombre_version = None
        self.notas = ''
        self.datamanager = None
        self.metadata = None

        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_demo = None
        self.y_demo = None
        self.metadata_split = None

        self.analisis_manifold = None
        self.modelo_feature_extraction = None

        self.scaler = None
        self.grid = None
        self.estimador = None

        self.individuo = None
        self.lista_individuos = []
        self.mejor_individuo_algoritmo_genetico = None
        self.resultados_algoritmo_genetico = pd.DataFrame()

        self.metrica_train = None
        self.metrica_test = None
        self.metrica_demo = None
        self.metrica_minima = None

        self.submission = None

        self.lista_modelos_seleccion_features = []

    # =========================================================================
    # GESTION DE ARCHIVOS
    # =========================================================================
    # Agrego X e y
    def marcar_nueva_version(self):
        self.nueva_version = True
        self.nombre_version = date2str(datetime.now())

    def marcar_vieja_version(self):
        self.nueva_version = False

    def modificar_carpeta(self, nueva_carpeta):
        self.carpeta = nueva_carpeta
        self.parser = Parser(self.nombre, self.carpeta)

    def agregar_datamanager(self, datamanager):
        if self.datamanager is not None:
            if not self.datamanager.X.columns.equals(datamanager.X.columns):
                self.marcar_nueva_version()
            else:
                self.marcar_vieja_version()
        else:
            self.marcar_nueva_version()

        # Cargo el dataset
        self.datamanager = datamanager
        self.agregar_dataset(self.datamanager.X, self.datamanager.y)

    def agregar_dataset(self, X, y):
        self.X = X
        self.y = y

    def agregar_nota(self, nota):

        if hasattr(self, 'notas'):
            self.notas = " ".join([self.notas, nota])
        else:
            self.notas = nota

    # Funciones de guardado
    def guardar(self, prefix="", suffix=""):

        # Actualizo la metadata
        self.calcular_metadata()

        # armo el nombre del archivo
        nombre_archivo = f"{prefix}{self.nombre}{suffix}"

        # Cargo el diccionario, borro el parser
        d_data = copy.copy(self.__dict__)
        del d_data["parser"]

        # Guardo
        self.parser.pickleo(d_data, nombre_archivo)

    def guardar_como(self, suffix=None):

        if suffix is None:
            suffix = self.nombre_version

        self.guardar(prefix="", suffix=suffix)

    # Funciones de carga
    def cargar(self, carpeta=None, nombre_archivo=None):
        # Si no defino la carpeta tomo la del objeto
        if not carpeta:
            carpeta = self.carpeta

        # Tomo el nombre del archivo
        if not nombre_archivo:
            nombre_archivo = f"{self.nombre}"

        # Cargo el objeto y genero el parser
        self.__dict__ = self.parser.unpickleo(nombre_archivo, carpeta=carpeta)
        self.__dict__["parser"] = Parser(nombre_archivo, carpeta)

    # Funciones de reseteo
    def resetear_individuos(self):
        if hasattr(self, 'lista_individuos'):
            self.lista_individuos = []

        if hasattr(self, 'resultados_algoritmo_genetico'):
            self.resultados_algoritmo_genetico = pd.DataFrame()

        if hasattr(self, 'mejor_individuo_algoritmo_genetico'):
            self.mejor_individuo_algoritmo_genetico = None

    def resetear_estimacion(self):
        if hasattr(self, 'grid'):
            self.grid = None

    def resetear_split(self):
        self.X_train = None
        self.X_test = None
        self.X_demo = None

    def reiniciar(self):
        self.resetear_individuos()
        self.resetear_estimacion()
        self.resetear_split()

    # =========================================================================
    # SPLIT SERIES
    # =========================================================================
    def split_series(
            self,
            funcion_split_train_test=None,
            parametros_split_train_test=None,
            funcion_split_test_demo=None,
            parametros_split_test_demo=None,
    ):
        """
        Método que permite cortar la serie en train, test y demo.
        Demo es una segunda muestra de X que sirve para un segundo testeo del
        analista.

        Parametros
        ----------
        funcion_split_train_test : funcion o string. Obligatorio.
            DESCRIPCION. Funcion que devuelve a los atributos X e y separados.
        La funcion debe devolver
            cuatro elementos: X_train, X_test, y_train e y_test

        Parametros_split_train_test: dict. Optativo. default dict(test_size=0.3,shuffle=False, random_state=0)
            DESCRIPCION. Diccionario de parametros para la funcion split.

        funcion_split_test_demo: idem funcion_split_train_test pero genera
        X_demo e y_demo desde X_test e y_test.
            DESCRIPCION. Si es None no se genera el split.

        parametros_split_test_demo
            DESCRIPCION. idem parametros_split_train_test para X_test y X_demo.

        """
        d = {}
        # Completo los datos si son None
        if not funcion_split_train_test:
            d.update(dict(funcion_split_train_test=train_test_split))
        elif isinstance(funcion_split_train_test, str):
            exec(f'd.update(dict(funcion_split_train_test={funcion_split_train_test}))')
        else:
            d.update(dict(funcion_split_train_test=funcion_split_train_test))

        if not parametros_split_train_test:
            d.update(dict(parametros_split_train_test=dict(test_size=0.3, shuffle=False, random_state=0)))
        else:
            d.update(dict(parametros_split_train_test=parametros_split_train_test))

        if not funcion_split_test_demo:
            d.update(dict(funcion_split_test_demo = train_test_split))
        elif isinstance(funcion_split_test_demo, str):
            exec(f'd.update(funcion_split_test_demo = {funcion_split_test_demo})')
        else:
            d.update(dict(funcion_split_test_demo=funcion_split_test_demo))

        if not parametros_split_test_demo:
            d.update(dict(parametros_split_test_demo = dict(test_size=0.5, shuffle=False, random_state=0)))
        else:
            d.update(dict(parametros_split_test_demo=parametros_split_test_demo))

        funcion_split_train_test = d['funcion_split_train_test']
        parametros_split_train_test = d['parametros_split_train_test']
        funcion_split_test_demo = d['funcion_split_test_demo']
        parametros_split_test_demo = d['parametros_split_test_demo']

        # Hago el split principal
        _X = self.X
        _y = self.y
        _params = parametros_split_train_test
        resultado = funcion_split_train_test(_X, _y, **_params)

        # Tomo los resultados
        self.X_train = resultado[0]
        self.X_test = resultado[1]
        self.y_train = resultado[2]
        self.y_test = resultado[3]

        # Hago el split segundo
        _X = self.X_test
        _y = self.y_test
        _params = parametros_split_test_demo
        resultado_demo = funcion_split_test_demo(_X, _y, **_params)

        # Tomo los resultados
        self.X_test = resultado_demo[0]
        self.X_demo = resultado_demo[1]
        self.y_test = resultado_demo[2]
        self.y_demo = resultado_demo[3]

        # Armo la metadata
        a = f'{funcion_split_train_test}'
        b = f'{parametros_split_train_test}'
        c = f'{funcion_split_test_demo}'
        d = f'{parametros_split_test_demo}'
        self.metadata_split = ' '.join([a, b, c, d])

    # =============================================================================
    # SCALER
    # =============================================================================
    def agregar_scaler(self, scaler=None, parametros_scaler=None):
        """
        Funcion que sirve cambiar la escala de X.

        Parametros
        ----------
        scaler: sklearn.scaler. Obligatorio.
            DESCRIPCION. Funcion scaler de sklearn que va a utilizarse para
            modificar la escala de los features.

        parametros_scaler: dict. Optativo. default dict()
            DESCRIPCION. Diccionario de parametros del scaler.

        """
        # Completo los valores por default
        if not scaler:
            scaler = MinMaxScaler

        if not parametros_scaler:
            parametros_scaler = {}

        # Cambio las dimensiones si el scaler es quantile
        if scaler == QuantileTransformer or scaler == "QuantileTransformer":
            d = dict(n_quantiles=min(1000, self.X_train.shape[0]))
            parametros_scaler.update(d)

        # Copio el scaler y hago el ajuste
        if isinstance(scaler, str):
            exec(f'from sklearn.preprocessing import {scaler}')
            a = f"self.scaler = {scaler}(**parametros_scaler)"
            b = ".fit(X=self.X_train)"
            string = "".join([a, b])
            exec(string)

        elif isinstance(scaler, type):
            self.scaler = scaler(**parametros_scaler).fit(X=self.X_train)

        else:
            if type(scaler) == QuantileTransformer:
                scaler.n_quantiles = min(1000, self.X_train.shape[0])
            self.scaler = scaler.fit(X=self.X_train)

        # Aplico las transformaciones
        _X_tr_tmp = self.scaler.transform(self.X_train)
        _X_te_tmp = self.scaler.transform(self.X_test)
        _X_de_tmp = self.scaler.transform(self.X_demo)

        # Armo el DataFrame train
        index_tr = self.X_train.index
        columns_tr = self.X_train.columns
        _X_tr_tmp = pd.DataFrame(_X_tr_tmp, index=index_tr, columns=columns_tr)

        # Armo el DataFrame test
        index_te = self.X_test.index
        columns_te = self.X_test.columns
        _X_te_tmp = pd.DataFrame(_X_te_tmp, index=index_te, columns=columns_te)

        # Armo el DataFrame demo
        index_de = self.X_demo.index
        columns_de = self.X_demo.columns
        _X_de_tmp = pd.DataFrame(_X_de_tmp, index=index_de, columns=columns_de)

        # Actualizo los datasets
        self.X_train = _X_tr_tmp
        self.X_test = _X_te_tmp
        self.X_demo = _X_de_tmp

    # =========================================================================
    # AJUSTES
    # =========================================================================
    def ajustar_cv(
            self,
            estimador,
            parametros_para_cruzar,
            parametros_grid=None,
            parametros_fit=None,
            tipo_cv=None,
    ):
        """
        Metodo que hace un ajuste por validacion cruzada.

        Parametros
        ----------
        estimador : dict. Obligatorio.
            DESCRIPCION. Es el estimador con el que se van a ajustar las
            series. debe ser compatible con sklearn.

        parametros_para_cruzar: dict. Obligatorio
            DESCRIPCION. Diccionario con los parametros para cruzar en el CV.
            Las keys
            del diccionario son parametros del estimador y los values son
            listas con distintos valores que van a usarse para calcular
            las distintas combinaciones de estos.

        parametros_grid : diccionario. Optativo. default: None
            DESCRIPCION. Diccionario con los parametros por defecto para el
            estimador
            grid. En caso de no pasarle ningun parametro adicional, el
            grid va a ajustar siguiendo los paramatros que le pase de
            manera fija para este modelo.

        parametros_fit : diccionario. Optativo. default: None
            DESCRIPCION. Diccionario con los parametros por defecto que van a
            usarse en
            el fit del estimador.

        tipo_cv. string : Optativo. default: None {None, 'grid', 'random'}
            DESCRIPCION. string que determina que tipo de grid voy a usar.
            Puede tomar
            el valor 'grid' para GridSearchCV o 'random' para
            RandomizedSearchCV.

        """
        # Completo los parametros si son None
        if not parametros_grid:
            parametros_grid = dict()

        if not parametros_fit:
            parametros_fit = dict()

        if not tipo_cv:
            tipo_cv = "grid"

        # Storeo el estimador en caso de ser una funcion o un string
        if type(estimador) == str:
            exec(f"self.estimador = {estimador}")
        else:
            self.estimador = estimador

        # Ajusto el modelo con los parametros para cruzar y los del grid
        if tipo_cv == "grid":
            self.grid = GridSearchCV(
                self.estimador, parametros_para_cruzar, **parametros_grid
            )
        elif tipo_cv == "random":
            self.grid = RandomizedSearchCV(
                self.estimador, parametros_para_cruzar, **parametros_grid
            )
        else:
            raise ValueError("Error en la definicion de tipo_cv")
        # Hago el ajuste
        _X = self.X_train.values
        _y = self.y_train.values
        self.grid.fit(_X, _y, **parametros_fit)

    # =========================================================================
    # Agrego una nueva serie test
    # =========================================================================
    def agregar_test(self, X_true, y_true=None, usar_scaler=True):
        """
        Metodo con el que se agrega una serie test al analista.

        Parametros
        ----------
        X_true: Pandas Dataframe. Obligatorio.
            DESCRIPCION. Dataframe que reemplaza al X_test.

        y_true: Pandas Series. Optativo. Default: None
            DESCRIPCION. Serie que reemplaza al y_test.

        usar_scaler: bool. Optativo. Default: True
            DESCRIPCION. Parámetro que define si en caso de existir se usa el
            scaler.

        """
        # Agrego el test viejo a la serie vieja
        self.X_test = X_true
        self.y_test = y_true

        # Aplico el scaler
        if hasattr(self, "scaler") and usar_scaler:
            # Aplico el scaler
            _X_transf = self.scaler.transform(self.X_test)

            # Convierto a DataFrame
            ind = self.X_test.index
            cols = self.X_test.columns
            _X_transf = pd.DataFrame(_X_transf, index=ind, columns=cols)
            self.X_test = _X_transf

        # Emparejo las columnas de X_train
        self.X_test = self.X_test.loc[:, self.X_train.columns]

    def predecir(
            self,
            usar_train=True,
            usar_test=True,
            usar_demo=True,
            calcular_proba=True,
            agregar_true=True,
    ):
        """
        Metodo que permite predecir en base a un modelo ajustado.

        Parametros
        ----------
        usar_train: bool, opcional. Default: True
            DESCRIPCION. Parametro que determina si se agrega X_train e y_train
            al diccionario resultado.

        usar_test: bool, opcional. Default: True
            DESCRIPCION. Parametro que determina si se agrega X_test e y_test
            al diccionario resultado.

        usar_demo: bool, opcional. Default: True
            DESCRIPCION.  Parametro que determina si se agrega X_demo e y_demo
            al diccionario resultado.

        calcular_proba: bool, opcional. Default: True
            DESCRIPCION.  Parametro que determina si se agregan los vectores
            de probabilidades al diccionario resultado.

        agregar_true: bool, opcional. Default: True
            DESCRIPCION. Parametro que determina si se agrega X_true al
            diccionario resultado.

        Returns
        -------
        d_resultado : dict
            DESCRIPCION. Diccionario con los resultados

        """
        # Creo el dict resultado
        d_resultado = {}

        def calc(_nombre, _X, _y, _calcular_proba, _agregar_true):

            # Creo el resultado
            d_res = {}

            # Calculo las predicciones
            index = _X.index
            _y_pred = self.grid.predict(_X.values)
            _y_pred = pd.Series(_y_pred, index=index)
            d_res[f"y_predict_{_nombre}"] = _y_pred

            # Si corresponde calculo la proba
            if calcular_proba:
                _y_proba = self.grid.predict_proba(_X.values)
                _y_proba = pd.DataFrame(_y_proba, index=index)

                d_res[f"y_predict_proba_{_nombre}"] = _y_proba

            # Si corresponde agrego el true
            if agregar_true:
                d_res[f"y_true_{_nombre}"] = _y

            # Devuelvo el resultado
            return d_res

        # Si corresponde train
        if usar_train:
            d_resultado.update(
                calc(
                    _nombre="train",
                    _X=self.X_train,
                    _y=self.y_train,
                    _calcular_proba=calcular_proba,
                    _agregar_true=agregar_true,
                )
            )
        if usar_test:
            d_resultado.update(
                calc(
                    _nombre="test",
                    _X=self.X_test,
                    _y=self.y_test,
                    _calcular_proba=calcular_proba,
                    _agregar_true=agregar_true,
                )
            )
        if usar_demo:
            d_resultado.update(
                calc(
                    _nombre="demo",
                    _X=self.X_demo,
                    _y=self.y_demo,
                    _calcular_proba=calcular_proba,
                    _agregar_true=agregar_true,
                )
            )
        return d_resultado

    # %%=========================================================================
    # METODOS DE AJUSTE AUTOMATIZADO
    # =========================================================================
    def analizar(
            self,
            splitear=None,
            funcion_split_train_test=None,
            parametros_split_train_test=None,
            funcion_split_test_demo=None,
            parametros_split_test_demo=None,
            scalar=None,
            scaler=None,
            parametros_scaler=None,
            ajustar=None,
            estimador=None,
            tipo_cv=None,
            parametros_para_cruzar=None,
            parametros_grid=None,
            parametros_fit=None,
            verbose=None,
    ):
        """
        Función que permite realizar el analisis del analista.

        Parametros
        ----------
        params_split : dict. Optativo. Default: None -> dict(test_size=0.7, shuffle=False)
            Diccionario con el que se configura el train_test_split, de acuerdo
            a la documentacion de train_test_set de la clase Sklearn.

        scaler : funcion scaler sklearn o string. Optativo. Default: None
            Scaler sklearn que va a aplicarse al dataset.

        params_scaler : dict. Optativo. default: None -> dict()
            Diccionario con los parametros que van a pasarse al scaler.

        estimador : estimador

        tipo_cv : string. Optativo. Default: None -> 'grid'.
            Parametro con el que se elije el tipo de cross validation que se
            va a aplicar. puede ser 'grid' para aplicarse GridSearchCV, o
            'random' si se aplica RandomizedSearchCV.

        parametros_grid : dict. Optativo. Default: None
            Default: dict(n_jobs=-1, scoring='precision', verbose=10, cv=StratifiedKFold(n_splits=5, shuffle=False))
            Diccionario con le que se parametriza el modelo de cross-
            validation seleccionado ('grid 'o 'random'), de acuerdo a la docu-
            mentacion de sklearn.

        parametros_para_cruzar : dict. Optativo. Default: None -> dict().
            Diccionario para el grid. Si es un diccionario vacio (valor por
            default) se toman los paramtros del estimador madre y se realiza
            el ajuste de este con esos parametros.

        """
        # Completo los parametros por defecto cuando son None
        if splitear is None:
            splitear = True

        if funcion_split_train_test is None:
            funcion_split_train_test = train_test_split

        if parametros_split_train_test is None:
            parametros_split_train_test = dict(test_size=0.3, shuffle=False)

        if funcion_split_test_demo is None:
            funcion_split_test_demo = train_test_split

        if parametros_split_test_demo is None:
            parametros_split_test_demo = dict(test_size=0.5, shuffle=False)

        if scalar is None:
            scalar = True

        if scaler is None:
            scaler = MinMaxScaler()

        if parametros_scaler is None:
            parametros_scaler = dict()

        if ajustar is None:
            ajustar = True

        if estimador is None:
            estimador = LGBMClassifier(max_bin=100, num_leaves=8, random_state=0)

        if tipo_cv is None:
            tipo_cv = "grid"

        if parametros_para_cruzar is None:
            parametros_para_cruzar = dict()

        if parametros_grid is None:
            parametros_grid = dict(verbose=0, cv=StratifiedKFold(n_splits=10, shuffle=False))

        if parametros_fit is None:
            parametros_fit = dict()

        if verbose is None:
            verbose = False

        # SPLITEO
        if splitear:
            if verbose:
                print("Spliteando las serie")

            self.split_series(
                funcion_split_train_test,
                parametros_split_train_test,
                funcion_split_test_demo,
                parametros_split_test_demo,
            )

        # SCALER
        if scalar and scaler is not None:
            if verbose:
                print("Agregando Scaler")
            self.agregar_scaler(scaler, **parametros_scaler)

        # AJUSTE POR CROSS VALIDATION
        if ajustar:
            if verbose:
                print("Haciendo el ajuste por CV")

            self.ajustar_cv(
                estimador,
                parametros_para_cruzar,
                parametros_grid,
                parametros_fit,
                tipo_cv,
            )

    def estimar(
            self,
            dict_estimador,
            splitear=None,
            funcion_split_train_test=None,
            parametros_split_train_test=None,
            funcion_split_test_demo=None,
            parametros_split_test_demo=None,
            scalar=None,
            scaler=None,
            parametros_scaler=None,
            tipo_cv=None,
            parametros_grid=None,
            parametros_fit=None,
            funcion_scorer=None,
            parametros_scorer=None,
            verbose=None,
    ):

        if funcion_scorer is None:
            funcion_scorer = 'accuracy_score'

        if parametros_scorer is None:
            parametros_scorer = dict()

        if parametros_grid is None:
            parametros_grid = dict(verbose=0, cv=StratifiedKFold(n_splits=10, shuffle=False))

        # instancio la metrica inicial
        metrica = 0

        # Corro para cada elemento de la lista
        # parametros_para_cruzar = dict_estimador['lista_param_grid'][0]
        for parametros_para_cruzar in dict_estimador["lista_param_grid"]:

            # Corro el pipeline
            if type(funcion_scorer) is str or type(funcion_scorer) is np.str_:

                string = "".join(
                    [
                        "parametros_grid.update(scoring=make_scorer",
                        f"({funcion_scorer}, **parametros_scorer))",
                    ]
                )
                exec(string)

            else:
                parametros_grid.update(
                    scoring=make_scorer(funcion_scorer, **parametros_scorer)
                )

            self.analizar(
                splitear=splitear,
                funcion_split_train_test=funcion_split_train_test,
                parametros_split_train_test=parametros_split_train_test,
                funcion_split_test_demo=funcion_split_test_demo,
                parametros_split_test_demo=parametros_split_test_demo,
                scalar=scalar,
                scaler=scaler,
                parametros_scaler=parametros_scaler,
                ajustar=True,
                estimador=dict_estimador["estimator"],
                tipo_cv=tipo_cv,
                parametros_para_cruzar=parametros_para_cruzar,
                parametros_grid=parametros_grid,
                parametros_fit=parametros_fit,
                verbose=verbose,
            )

            # Calculo la metrica sobre el test
            self.metrica_train = self.grid.best_score_

            self.metrica_test = self.calcular_metrica(
                funcion_scorer=funcion_scorer,
                usar_train=False,
                usar_test=True,
                usar_demo=False,
                parametros_scorer=parametros_scorer,
            )

            self.metrica_demo = self.calcular_metrica(
                funcion_scorer,
                usar_train=False,
                usar_test=False,
                usar_demo=True,
                parametros_scorer=parametros_scorer,
            )

            _m_tr = self.metrica_train
            _m_te = self.metrica_test
            _m_de = self.metrica_demo
            self.metrica_minima = min(_m_tr, _m_te, _m_de)

            if self.metrica_test >= metrica:
                _best_params = self.grid.best_params_
                dict_estimador["estimator"].__dict__.update(_best_params)
                metrica = self.metrica_test

    def estimar_por_lista_dict_estimador(
            self,
            lista_dict_estimador,
            splitear=None,
            funcion_split_train_test=None,
            parametros_split_train_test=None,
            funcion_split_test_demo=None,
            parametros_split_test_demo=None,
            scalar=None,
            scaler=None,
            parametros_scaler=None,
            tipo_cv=None,
            parametros_grid=None,
            parametros_fit=None,
            funcion_scorer=None,
            parametros_scorer=None,
            verbose=None,
    ):

        # Hago el spliteo
        self.analizar(
            splitear=splitear,
            funcion_split_train_test=funcion_split_train_test,
            parametros_split_train_test=parametros_split_train_test,
            funcion_split_test_demo=funcion_split_test_demo,
            parametros_split_test_demo=parametros_split_test_demo,
            scalar=scalar,
            scaler=scaler,
            parametros_scaler=parametros_scaler,
            ajustar=False,
            tipo_cv=tipo_cv,
            parametros_grid=parametros_grid,
            parametros_fit=parametros_fit,
            verbose=verbose,
        )

        # dict_estimador = lista_dict_estimador[0]
        for dict_estimador in lista_dict_estimador:
            # Ajusto
            self.estimar(
                dict_estimador,
                splitear=False,
                scalar=False,
                funcion_split_train_test=funcion_split_train_test,
                parametros_split_train_test=parametros_split_train_test,
                funcion_split_test_demo=funcion_split_test_demo,
                parametros_split_test_demo=parametros_split_test_demo,
                parametros_scaler=parametros_scaler,
                tipo_cv=tipo_cv,
                parametros_grid=parametros_grid,
                parametros_fit=parametros_fit,
                funcion_scorer=funcion_scorer,
                parametros_scorer=parametros_scorer,
                verbose=verbose,
            )

    def ajustar_por_algoritmo_genetico(
            self,
            lista_estimadores,
            agregar_hiperparametros=None,
            agregar_total_hiperparametros=None,  # <---- TODOS LOS PARAMETROS
            filtrar_scaler=None,  # <---- EL SCALER
            filtrar_columnas=None,  # <---- LAS COLUMNAS
            probabilidad_ocurrencia_columna=None,
            filtrar_filas=None,  # <---- LAS FILAS
            probabilidad_ocurrencia_fila=None,
            dd_parametros_adicionales=None,
            parametros_algo=None,
    ):

        # Completo los parametros por defecto
        if agregar_hiperparametros is None:
            agregar_hiperparametros = True

        if agregar_total_hiperparametros is None:
            agregar_total_hiperparametros = True  # <---- TODOS LOS PARAMETROS

        if filtrar_scaler is None:
            filtrar_scaler = False  # <---- EL SCALER

        if filtrar_columnas is None:
            filtrar_columnas = True  # <---- LAS COLUMNAS

        if probabilidad_ocurrencia_columna is None:
            probabilidad_ocurrencia_columna = 0.5

        if filtrar_filas is None:
            filtrar_filas = True  # <---- LAS FILAS

        if probabilidad_ocurrencia_fila is None:
            probabilidad_ocurrencia_fila = 0.5

        if dd_parametros_adicionales is None:
            dd_parametros_adicionales = None

        if parametros_algo is None:
            parametros_algo = dict(
                n_population=50,
                n_generations=200,
                crossover_proba=[0.5, 1.0],
                mutation_proba=[0.5, 1.0],
                crossover_independent_proba=[0.05, 0.3],
                mutation_independent_proba=[0.05, 0.3],
                tournament_size=3,
                n_random_population_each_gen=10,
                add_mutated_hall_of_fame=True,
                n_gen_no_change=20,
            )

        # Creo los objetos si no existen
        if not hasattr(self, "lista_individuos"):
            setattr(self, "lista_individuos", list())

        if not hasattr(self, "resultados_algoritmo_genetico"):
            setattr(self, "resultados_algoritmo_genetico", pd.DataFrame())

        # Armo el diccionario con los parametros
        if hasattr(self, "X_train"):
            _X = self.X_train

        else:
            _X = self.X

        d_total = generar_parametros_algoritmo_genetico(
            X_analisis=_X,
            lista_estimadores=lista_estimadores,
            agregar_hiperparametros=agregar_hiperparametros,
            agregar_total_hiperparametros=agregar_total_hiperparametros,
            filtrar_scaler=filtrar_scaler,  # <---- EL SCALER
            filtrar_columnas=filtrar_columnas,  # <---- LAS COLUMNAS
            probabilidad_ocurrencia_columna=probabilidad_ocurrencia_columna,
            filtrar_filas=filtrar_filas,  # <---- LAS FILAS
            probabilidad_ocurrencia_fila=probabilidad_ocurrencia_fila,
            dd_parametros_adicionales=dd_parametros_adicionales,
        )

        # Armo la Funcion
        def ajustar_individuo(individuo):

            # Copio los objetos
            self2 = copy.copy(self)
            self2.estimar_por_individuo(individuo)

            # Adjunto el individuo
            self.lista_individuos.append(individuo)

            # Adjunto los resultados
            res = pd.DataFrame(
                [
                    dict(
                        metrica_train=self2.metrica_train,
                        metrica_test=self2.metrica_test,
                        metrica_demo=self2.metrica_demo,
                    )
                ]
            )

            self.resultados_algoritmo_genetico = pd.concat(
                [self.resultados_algoritmo_genetico, res]
            )

            return self2.metrica_minima

        # Armo el algoritmo
        algo = AlgoritmoGenetico(
            dict_variables=d_total,
            function=ajustar_individuo,
            correr_algortimo=False,
            **parametros_algo,
        )

        # individuos = algo.inicializar_poblacion(cantidad=20)
        # individuo = individuos[2]

        # Corro el algoritmo
        algo.correr_algortimo()

        # Calculo las metricas
        self.mejorar_metricas_algoritmo_genetico()

        # Storeo el mejor individuo del algo
        self.mejor_individuo_algoritmo_genetico = algo.mejor_individuo

    def mejorar_metricas_algoritmo_genetico(self):

        # Acomodo los resultados
        self.resultados_algoritmo_genetico.reset_index(drop=True, inplace=True)

        # Calculo las metricas adicionales
        a = self.resultados_algoritmo_genetico
        _metricas = a[["metrica_train", "metrica_test", "metrica_demo"]]
        a["metrica_minima"] = _metricas.min(axis=1)
        a["metrica_media"] = _metricas.mean(axis=1)

        # Calculo los quantiles
        def calcular_ranking(df2, nombre_metrica, pct):
            _a = df2[f'metrica_{nombre_metrica}']
            _a = _a.rank(method='dense', ascending=False, pct=pct)
            return _a

        df = self.resultados_algoritmo_genetico

        # rankings
        a['ranking_metrica_train'] = calcular_ranking(df2=df, nombre_metrica='train', pct=False)
        a['ranking_metrica_test'] = calcular_ranking(df2=df, nombre_metrica='test', pct=False)
        a['ranking_metrica_demo'] = calcular_ranking(df2=df, nombre_metrica='demo', pct=False)
        a['ranking_metrica_minima'] = calcular_ranking(df2=df, nombre_metrica='minima', pct=False)
        a['ranking_metrica_media'] = calcular_ranking(df2=df, nombre_metrica='media', pct=False)

        # quantiles
        a['quantile_metrica_train'] = calcular_ranking(df2=df, nombre_metrica='train', pct=True)
        a['quantile_metrica_test'] = calcular_ranking(df2=df, nombre_metrica='test', pct=True)
        a['quantile_metrica_demo'] = calcular_ranking(df2=df, nombre_metrica='demo', pct=True)
        a['quantile_metrica_minima'] = calcular_ranking(df2=df, nombre_metrica='minima', pct=True)
        a['quantile_metrica_media'] = calcular_ranking(df2=df, nombre_metrica='media', pct=True)

        # Storeo el resultado
        self.resultados_algoritmo_genetico = a

    # METODOS PARA TRABAJAR CON INDIVIDUOS
    def generar_analista_estimado_por_individuo(self, individuo):

        self_copy = copy.deepcopy(self)
        self_copy.estimar_por_individuo(individuo)

        return self_copy

    # Armo la Funcion
    def estimar_por_individuo(self, individuo):

        # Storeo
        self.individuo = individuo

        # Tomo el scaler
        individuo_parametros = {
            k.replace("params__", ""): v
            for k, v in individuo.items()
            if "params__" in k
        }

        # Parametros por defecto
        # Split
        if "funcion_split_train_test" not in individuo_parametros.keys():
            individuo_parametros["funcion_split_train_test"] = train_test_split

        if "parametros_split_train_test" not in individuo_parametros.keys():
            d = dict(test_size=0.3, shuffle=False, random_state=0)
            individuo_parametros["parametros_split_train_test"] = d

        if "funcion_split_test_demo" not in individuo_parametros.keys():
            individuo_parametros["funcion_split_test_demo"] = train_test_split

        if "parametros_split_test_demo" not in individuo_parametros.keys():
            d = dict(test_size=0.5, shuffle=False, random_state=0)
            individuo_parametros["parametros_split_test_demo"] = d

        # Scaler
        if "parametros_scaler" not in individuo_parametros.keys():
            individuo_parametros["parametros_scaler"] = dict()

        # Ajuste grid
        if "ajustar" not in individuo_parametros.keys():
            individuo_parametros["ajustar"] = True

        if "tipo_cv" not in individuo_parametros.keys():
            individuo_parametros["tipo_cv"] = "grid"

        if "parametros_grid" not in individuo_parametros.keys():
            cv = StratifiedKFold(n_splits=10, shuffle=False)
            d = dict(n_jobs=-1, verbose=0, cv=cv)
            individuo_parametros["parametros_grid"] = d

        if "parametros_fit" not in individuo_parametros.keys():
            individuo_parametros["parametros_fit"] = dict()

        # Scoring
        if "funcion_scorer" not in individuo_parametros.keys():
            individuo_parametros["funcion_scorer"] = accuracy_score

        if "parametros_scorer" not in individuo_parametros.keys():
            individuo_parametros["parametros_scorer"] = dict()

        # Otros
        if "verbose" not in individuo_parametros.keys():
            individuo_parametros["verbose"] = False

        # Tomo el scaler
        individuo_scaler = {
            k.replace("scaler__", ""): v
            for k, v in individuo.items()
            if "scaler__" in k
        }

        if len(individuo_scaler) > 0:
            splitear = True
            scalar = True
            scaler = individuo_scaler["scaler"]
        else:
            splitear = False
            scalar = False
            scaler = None
        # Tomo el estimador
        individuo_estimador = {
            k.replace("estimador__", ""): v
            for k, v in individuo.items()
            if "estimador__" in k
        }

        # Tomo el nombre de la funcion
        funcion_estimador = individuo_estimador["funcion_estimador"]

        if type(funcion_estimador) is np.str_ or type(funcion_estimador) is str:
            nombre_funcion2 = funcion_estimador

        else:
            nombre_funcion = f"""'{individuo_estimador['funcion_estimador']}"""
            nombre_funcion = nombre_funcion[9:-2]
            nombre_funcion2 = nombre_funcion.split(".")[-1]

        # Tomo los Hiperparametros
        d_h = {
            k.replace(f"""hiperparametros__{nombre_funcion2}_""", ""): v
            for k, v in individuo.items()
            if "hiperparametros__" in k and nombre_funcion2 in k
        }
        individuo_hiperparametros = d_h

        if len(individuo_hiperparametros) > 0:
            lista_param_grid = {k: [v] for k, v in individuo_hiperparametros.items()}
            lista_dict_estimador = [
                generar_estimador(
                    funcion_estimador=funcion_estimador,
                    lista_param_grid=[lista_param_grid],
                )
            ]

        else:
            lista_dict_estimador = list()
            lista_dict_estimador.extend(
                [
                    generar_estimador(
                        funcion_estimador=funcion_estimador, lista_param_grid=None
                    )
                ]
            )

        # Tomo las Columnas
        individuo_columnas = {
            k.replace("columnas__", ""): v
            for k, v in individuo.items()
            if "columnas__" in k
        }

        if len(individuo_columnas) > 0:

            ind_columnas = list(individuo_columnas.values())

            # Si escalo entonces saco las columnas de X, si no desde el train y el test
            self.X = self.X.loc[:, ind_columnas]
            if hasattr(self, "X_train"):
                self.X_train = self.X_train.iloc[:, ind_columnas]
                self.X_test = self.X_test.iloc[:, ind_columnas]
                self.X_demo = self.X_demo.iloc[:, ind_columnas]

        # Estimo
        self.analizar(
            splitear=splitear,
            funcion_split_train_test=individuo_parametros["funcion_split_train_test"],
            parametros_split_train_test=individuo_parametros[
                "parametros_split_train_test"
            ],
            funcion_split_test_demo=individuo_parametros["funcion_split_test_demo"],
            parametros_split_test_demo=individuo_parametros[
                "parametros_split_test_demo"
            ],
            scalar=scalar,
            scaler=scaler,
            ajustar=False,
            parametros_scaler=individuo_parametros["parametros_scaler"],
            tipo_cv=individuo_parametros["tipo_cv"],
            parametros_grid=individuo_parametros["parametros_grid"],
            parametros_fit=individuo_parametros["parametros_fit"],
            verbose=individuo_parametros["verbose"],
        )

        # Tomo las Filas
        individuo_filas = {
            k.replace("filas__", ""): v for k, v in individuo.items() if "filas__" in k
        }

        if len(individuo_filas) > 0:

            # Saco las filas desde X, si no desde el train y el test
            filas_total = pd.Series(individuo_filas)
            _X_train_ind_strings = self.X_train.index.map(str)
            filas_train = filas_total.filter(items=_X_train_ind_strings)

            self.X_train = self.X_train.loc[list(filas_train), :]
            self.y_train = self.y_train.loc[list(filas_train)]

            if hasattr(self, "X_demo"):
                self.X = pd.concat([self.X_train, self.X_test, self.X_demo])
                self.X = self.X.sort_index()
            else:
                self.X = pd.concat([self.X_train, self.X_test]).sort_index()

        for dict_estimador in lista_dict_estimador:  # dict_estimador = lista_dict_estimador[0]

            # Ajusto
            self.estimar(
                dict_estimador,
                splitear=False,
                funcion_split_train_test=individuo_parametros[
                    "funcion_split_train_test"
                ],
                parametros_split_train_test=individuo_parametros[
                    "parametros_split_train_test"
                ],
                funcion_split_test_demo=individuo_parametros["funcion_split_test_demo"],
                parametros_split_test_demo=individuo_parametros[
                    "parametros_split_test_demo"
                ],
                scalar=False,
                scaler=scaler,
                parametros_scaler=individuo_parametros["parametros_scaler"],
                tipo_cv=individuo_parametros["tipo_cv"],
                parametros_grid=individuo_parametros["parametros_grid"],
                parametros_fit=individuo_parametros["parametros_fit"],
                funcion_scorer=individuo_parametros["funcion_scorer"],
                parametros_scorer=individuo_parametros["parametros_scorer"],
                verbose=individuo_parametros["verbose"],
            )

    def predecir_por_individuo(self,
                               individuo,
                               usar_train=True,
                               usar_test=True,
                               usar_demo=True,
                               calcular_proba=True):

        self2 = self.generar_analista_estimado_por_individuo(individuo)

        d_preds = self2.predecir(
            usar_train=usar_train,
            usar_test=usar_test,
            usar_demo=usar_demo,
            calcular_proba=calcular_proba
        )

        return d_preds

    def predecir_individuos_por_filtro(
            self,
            d_filtros,
            usar_train=True,
            usar_test=True,
            usar_demo=True,
            calcular_proba=True,
    ):

        # Filtro el índice segun los filtros
        if d_filtros:
            d = {f"{a[0]}": [a[2], a[1]] for a in d_filtros}
            _df = filtrar_df(df=self.resultados_algoritmo_genetico, diccionario=d)
            index = _df.index
        else:
            index = self.resultados_algoritmo_genetico.index

        # Tomo la lista de individuos
        lista_individuos = [self.lista_individuos[ind] for ind in index]

        # Armo el dict resultado y los parametros
        d = {}
        _params = dict(usar_train=usar_train,
                       usar_test=usar_test,
                       usar_demo=usar_demo,
                       calcular_proba=calcular_proba)

        # Calculo las predicciones individuo = ind = lista_individuos[0]
        preds = [self.predecir_por_individuo(ind, **_params) for ind in lista_individuos]

        # asigno el train test y demo al diccionario resultado si corresponde
        if usar_train:
            pr_tr = pd.concat([pr["y_predict_train"] for pr in preds], axis=1)
            pr_tr.columns = index
            d.update(dict(y_predict_train=pr_tr))

        if usar_test:
            pr_te = pd.concat([pr["y_predict_test"] for pr in preds], axis=1)
            pr_te.columns = index
            d.update(dict(y_predict_test=pr_te))

        if usar_demo:
            pr_de = pd.concat([pr["y_predict_demo"] for pr in preds], axis=1)
            pr_de.columns = index
            d.update(dict(y_predict_demo=pr_de))

        if calcular_proba:

            # Calculo las probas
            def concat_probas(preds2, nombre='train'):

                # Armo el dict resultado
                _d = {}

                # Concateno las probas del mismo nombre
                _pr_pro = pd.concat([p[f"y_predict_proba_{nombre}"] for p in preds2], axis=1)
                preds_proba = _pr_pro

                # etiqueto
                for etiq in pd.unique(preds_proba.columns):
                    a = preds_proba[etiq]
                    a.columns = index
                    d.update({f"preds_proba_{nombre}_{etiq}": a})

                return _d

            # asigno el train test y demo al diccionario resultado si corresponde
            if usar_train:
                preds_proba_train = concat_probas(preds, nombre='train')
                d.update(preds_proba_train)

            if usar_test:
                preds_proba_test = concat_probas(preds, nombre='test')
                d.update(preds_proba_test)

            if usar_demo:
                preds_proba_demo = concat_probas(preds, nombre='demo')
                d.update(preds_proba_demo)

        return d

    def generar_submission(
            self,
            X_true,
            y_true=None,
            exportar=True,
            nombres_columnas=None,
            nombre_archivo="gender_submission",
    ):
        if nombres_columnas is None:
            nombres_columnas = ["PassengerId", "Survived"]

        # Chequeo que sea uno de los dos formatos
        if y_true is not None and X_true is not None:
            raise TypeError('Error en X_true e y_true. Solo puede cargarse un parametro.')

        # Agrego el test y hago la prediccion
        elif y_true is None and X_true is not None:

            self.agregar_test(X_true)

            d_preds = self.predecir(
                usar_train=False,
                usar_test=True,
                usar_demo=False,
                calcular_proba=False,
            )

            res = d_preds["y_predict_test"]

        else:
            res = y_true

        # Doy formato y storeo
        res = res.reset_index(drop=False).astype(int)
        res.columns = nombres_columnas
        self.submission = res

        # Exporto a csv
        if exportar:
            ruta_guardar = self.parser.carpeta + nombre_archivo + ".csv"
            res.to_csv(ruta_guardar, index=False, header=True)

    # %%=========================================================================
    # METRICAS
    # =========================================================================
    def calcular_metrica(
            self,
            funcion_scorer,
            usar_train=False,
            usar_test=True,
            usar_demo=False,
            parametros_scorer=None,
    ):
        if parametros_scorer is None:
            parametros_scorer=dict()

        # Chequeo que esté ajustado el modelo
        if not hasattr(self, "grid"):
            raise ValueError("Modelo sin ajustar")

        # Chequeo que solo haya uno
        assert usar_train + usar_test + usar_demo == 1

        # Calculo las predicciones
        d_predicciones = self.predecir(
            usar_train=usar_train,
            usar_test=usar_test,
            usar_demo=usar_demo,
            calcular_proba=False,
        )

        # Calculo la metrica
        if usar_train:
            y_true = self.y_train
            y_predict = d_predicciones["y_predict_train"]
        elif usar_test:
            y_true = self.y_test
            y_predict = d_predicciones["y_predict_test"]
        elif usar_demo:
            y_true = self.y_demo
            y_predict = d_predicciones["y_predict_demo"]
        else:
            raise TypeError("Error.")

        # Creo el dict resultado
        d = {}

        # Si es string o np.str ejecuto una cadena, si no ejecuto la funcion
        if type(funcion_scorer) is str or type(funcion_scorer) is np.str_:
            a = f"d.update(dict(metrica={funcion_scorer}(y_true, y_predict, **parametros_scorer)))"
            exec(a)

        else:
            a = dict(metrica=funcion_scorer(y_true, y_predict, **parametros_scorer))
            d.update(a)

        # retorno el resultado
        return d["metrica"]

    def calcular_metrica2(
            self,
            y_true,
            y_predict,
            funcion_scorer,
            parametros_scorer=None,
    ):

        # Completo el parametro
        if parametros_scorer is None:
            parametros_scorer = dict()

        # Creo el dict resultado
        d = {}

        # Si es string o np.str ejecuto una cadena, si no ejecuto la funcion
        if type(funcion_scorer) is str or type(funcion_scorer) is np.str_:
            a = f"d.update(dict(metrica={funcion_scorer}(y_true, y_predict, **parametros_scorer)))"
            exec(a)

        else:
            a = dict(metrica=funcion_scorer(y_true, y_predict, **parametros_scorer))
            d.update(a)

        # retorno el resultado
        return d["metrica"]

    def calcular_metricas_total(
        self,
        funcion_scorer
    ):

        d = {'train': self.calcular_metrica(
            funcion_scorer=funcion_scorer,
            usar_train=True,
            usar_test=False,
            usar_demo=False,
        ), 'test': self.calcular_metrica(
            funcion_scorer=funcion_scorer,
            usar_train=False,
            usar_test=True,
            usar_demo=False,
        ), 'demo': self.calcular_metrica(
            funcion_scorer=funcion_scorer,
            usar_train=False,
            usar_test=False,
            usar_demo=True,
        )}

        return d

    def estimar_por_funcion_estimador(
        self,
        funcion_estimador,
        lista_param_grid=None,
        **kwargs
    ):

        # Copio el objeto
        self2 = copy.deepcopy(self)

        # estimo
        lista_dict_estimador = [generar_estimador(funcion_estimador, lista_param_grid)]
        self2.estimar_por_lista_dict_estimador(lista_dict_estimador, **kwargs)

        return self2

    def calcular_features_importance(
        self,
        funcion_estimador='LGBMClassifier',
        lista_param_grid=None,
        **kwargs
    ):
        # Si no esta ajustado lo ajusto, sino copio
        if self.grid is None:
            # Estimo
            self2 = self.estimar_por_funcion_estimador(
                funcion_estimador=funcion_estimador,
                lista_param_grid=lista_param_grid,
                **kwargs
            )
        else:
            self2 = copy.deepcopy(self)

        # Calculo las importances
        if hasattr(self2.grid.best_estimator_, 'feature_importances_'):
            importances = self2.grid.best_estimator_.feature_importances_
        elif hasattr(self2.grid.best_estimator_, 'coef_'):
            importances = self2.grid.best_estimator_.coef_
        else:
            raise TypeError('Error en la funcion estimador')

        columns = self2.X_train.columns
        imp = pd.DataFrame([columns, importances]).T.sort_values(by=1, ascending=False)
        imp.set_index(0, inplace=True)
        imp.columns = [funcion_estimador]
        imp = imp.rank(method='min', pct=True)
        self.feature_importance = imp

    def eliminar_features(self, lista_features):

        # Copio el objeto
        self2 = copy.deepcopy(self)

        # Elimino del X
        self2.X = self2.X.drop(lista_features, axis=1)

        # elimino de las particiones solo si estan hechas
        if hasattr(self2, 'X_train'):
            self2.X_train = self2.X_train.drop(lista_features, axis=1)

        if hasattr(self2, 'X_test'):
            self2.X_test = self2.X_test.drop(lista_features, axis=1)

        if hasattr(self2, 'X_demo'):
            self2.X_demo = self2.X_demo.drop(lista_features, axis=1)

        # Marco como nueva version
        self2.marcar_nueva_version()
        self2.reiniciar()

        return self2

    def generar_modelo(
            self,
            nombre_funcion_modelo,
            nombre_libreria_origen,
            params_modelo=None
    ):

        if params_modelo is None:
            params_modelo = dict()

        # Importo la libreria
        exec(f"from {nombre_libreria_origen} import {nombre_funcion_modelo}")

        # Creo el selector
        m = {}
        exec(f"m.update(dict(modelo={nombre_funcion_modelo}(**params_modelo)))")

        return m['modelo']

    def fitear_modelo(
            self,
            nombre_funcion_modelo,
            nombre_libreria_origen,
            params_modelo=None,
            params_fit=None,
            nombre_X_fit=None,
            nombre_y_fit=None,
            transformar=False,
    ):
        # Genero el modelo
        modelo = self.generar_modelo(
            nombre_funcion_modelo=nombre_funcion_modelo,
            nombre_libreria_origen=nombre_libreria_origen,
            params_modelo=params_modelo
        )

        # Incorporo X e y en el dict para el ajuste
        d = {}
        if isinstance(nombre_X_fit, str):
            d.update(X=getattr(self, nombre_X_fit))
        else:
            d.update(X=nombre_X_fit)

        if isinstance(nombre_y_fit, str):
            d.update(y=getattr(self, nombre_y_fit))
        else:
            d.update(y=nombre_y_fit)

        if params_fit is not None:
            d.update(params_fit)

        # Ajusto
        if transformar:
            modelo2 = modelo.fit_transform(**d)
        else:
            modelo2 = modelo.fit(**d)

        return modelo2

    def aplicar_modelo_fiteado(
        self,
        modelo,
        funcion_wrapper_transform=None,
        lista_nombres_X_transform=None,
    ):

        if lista_nombres_X_transform is None:
            lista_nombres_X_transform = ['X', 'X_train', 'X_test', 'X_demo']

        # Transformo si corresponde
        cond1 = len(lista_nombres_X_transform) > 0
        cond2 = funcion_wrapper_transform is not None

        if cond1 and cond2:
            for attr in lista_nombres_X_transform:
                df = getattr(self, attr)
                df2 = funcion_wrapper_transform(modelo, df)
                setattr(self, attr, df2)

            self.marcar_nueva_version()

    def transformar_dataset(self, modelo, f):

        self.X = f(modelo, self.X)
        self.X_train = f(modelo, self.X_train)
        self.X_test = f(modelo, self.X_test)
        self.X_demo = f(modelo, self.X_demo)

    def calcular_metadata(self):

        # Creo el diccionario resultado
        meta = {}

        # Actualizo con los atributos
        lista_metadatos = [
            'nombre',
            'carpeta',
            'nombre_version',
            'notas',
            'metadata_split',
            'scaler',
            'grid',
            'metrica_train',
            'metrica_test',
            'metrica_demo',
            'metrica_minima',
            'notas',
        ]
        atributos = list(self.__dict__.keys())
        meta.update({meta: f'{getattr(self, meta)}' for meta in lista_metadatos if meta in atributos})

        # Actualizo con calculos
        def metadata_series(nombre_serie):

            if getattr(self, nombre_serie) is None:
                return dict()

            serie = getattr(self, nombre_serie)

            d = {}
            d.update({f'{nombre_serie}_filas': serie.shape[0]})

            if isinstance(serie, pd.DataFrame):
                d.update({f'{nombre_serie}_columnas': serie.shape[1]})
            else:
                d.update({f'{nombre_serie}_columnas': 1})

            return d

        # Actualizo las series
        meta.update(metadata_series(nombre_serie='X'))
        meta.update(metadata_series(nombre_serie='X_train'))
        meta.update(metadata_series(nombre_serie='X_test'))
        meta.update(metadata_series(nombre_serie='X_demo'))
        meta.update(metadata_series(nombre_serie='y'))
        meta.update(metadata_series(nombre_serie='y_train'))
        meta.update(metadata_series(nombre_serie='y_test'))
        meta.update(metadata_series(nombre_serie='y_demo'))

        # analizo si esta spliteado
        meta.update({'esta_spliteado': hasattr(self, 'X_train')})

        # Data_algoritmo_genetico
        if len(self.resultados_algoritmo_genetico) > 0:
            meta.update({'iter_algoritmo_genetico': len(self.resultados_algoritmo_genetico)})
            a2 = self.resultados_algoritmo_genetico.max(axis=0)
            a2 = a2.to_dict()
            a2 = {f'max_{k}_algoritmo_genetico': v for k, v in a2.items()}
            meta.update(a2)

        if hasattr(self, 'analisis_manifold') and getattr(self, 'analisis_manifold') is not None:
            meta.update(dict(silhouette = self.analisis_manifold['silhouette']))
            meta.update(dict(calinski_harabasz=self.analisis_manifold['calinski_harabasz']))
            meta.update(dict(davies_bouldin=self.analisis_manifold['davies_bouldin']))

        # Storeo
        self.metadata = meta

    def analizar_dataset_por_manifold(
        self,
        dataset,
        y=None,
        nombre_funcion_modelo='MDS',
        nombre_libreria_origen='sklearn.manifold',
        params_modelo=dict(n_components=2, max_iter=100, n_init=1, random_state=0),
        calcular_silhouette=True,
        calcular_calinski_harabasz=True,
        calcular_davies_bouldin=True,
        calcular_grafico=True,
        inplace=False,
    ):
        d = {}
        array_mds = self.fitear_modelo(
            nombre_funcion_modelo=nombre_funcion_modelo,
            nombre_libreria_origen=nombre_libreria_origen,
            params_modelo=params_modelo,
            params_fit=None,
            nombre_X_fit=dataset,
            nombre_y_fit=None,
            transformar=True,
        )

        # Adjunto al dict final
        d.update({'dataset': dataset})
        d.update({'data': pd.DataFrame(array_mds)})

        # Tomo el y que corresponde
        if y is None:
            if dataset == 'X':
                y = getattr(self, 'y')
            elif dataset == 'X_train':
                y = getattr(self, 'y_train')
            elif dataset == 'X_test':
                y = getattr(self, 'y_test')
            elif dataset == 'X_demo':
                y = getattr(self, 'y_demo')

        # Calculo el silouette
        if calcular_silhouette:
            d.update({'silhouette': silhouette_score(X=array_mds, labels=y)})

        if calcular_calinski_harabasz:
            d.update({'calinski_harabasz': calinski_harabasz_score(X=array_mds, labels=y)})

        if calcular_davies_bouldin:
            d.update({'davies_bouldin': davies_bouldin_score(X=array_mds, labels=y)})

        # Hago el grafico
        if calcular_grafico:
            df = pd.DataFrame(array_mds)
            df['color'] = y.reset_index(drop=True)
            d.update({'fig': px.scatter(df, y=1, x=0, color='color')})

        if inplace:
            self.analisis_manifold = d
        else:
            return d

    # FEATURE SELECTION
    def seleccionar_features_por_gars(
            self,
            cant_max_columnas,
            proba_apagar,
            parametros_algo=None,
    ):
        # Funcion
        nombre_funcion_modelo = 'MDS'
        nombre_libreria_origen = 'sklearn.manifold'
        params_modelo = dict(n_components=2, max_iter=100, n_init=1, random_state=0)

        def ajustar_individuo(individuo):

            ind_columnas = list(individuo.values())

            # Creo el diccionario con las columnas
            d = dict.fromkeys(ind_columnas)
            d = {k: v for (k, v) in d.items() if k > 0}

            # armo la lista final
            ind_columnas = list(d)

            # termino si son todas 0
            if len(ind_columnas) == 0:
                return 0.

            # X2 = copy.deepcopy(self.X_train)
            X2 = self.X_train.iloc[:, ind_columnas].copy(deep=True)

            array_mds = self.fitear_modelo(
                nombre_funcion_modelo=nombre_funcion_modelo,
                nombre_libreria_origen=nombre_libreria_origen,
                params_modelo=params_modelo,
                params_fit=None,
                nombre_X_fit=X2,
                nombre_y_fit=None,
                transformar=True,
            )

            # Silouette metrica
            sil2 = silhouette_score(X=array_mds, labels=self.y_train)

            return sil2

        # dict_variables
        cant_cols = self.X.shape[1]
        high = cant_cols - 1
        low = int(cant_cols * proba_apagar)
        d_total = {}
        for col in range(cant_max_columnas):
            d_total.update({f"col_{col}": dict(type=int, low=-low, high=high)})

        # parametros_algo
        if parametros_algo is None:
            parametros_algo = dict(
                n_population=50,
                n_generations=200,
                crossover_proba=0.8,
                mutation_proba=0.8,
                crossover_independent_proba=[0.05, 0.2],
                mutation_independent_proba=[0.05, 0.2],
                tournament_size=10,
                n_random_population_each_gen=5,
                add_mutated_hall_of_fame=True,
                n_gen_no_change=20,
            )

        # Creo el algoritmo
        algo = AlgoritmoGenetico(
            dict_variables=d_total,
            function=ajustar_individuo,
            correr_algortimo=False,
            **parametros_algo,
        )

        # Lo corro
        algo.correr_algortimo()

        d = {}
        d['algo'] = algo
        individuo = algo.mejor_individuo
        ind_columnas = list(individuo.values())

        # Creo el diccionario con las columnas
        d2 = dict.fromkeys(ind_columnas)
        d2 = {k: v for (k, v) in d2.items() if k > 0}
        ind_columnas = list(d2)
        d['support'] = ind_columnas

        return d

    def transformar_dataset_por_modelo(
            self,
            nombre_funcion_modelo,
            nombre_libreria_origen,
            params_modelo,
            params_fit,
            funcion_transformacion,
    ):
        modelo = self.fitear_modelo(
            nombre_funcion_modelo=nombre_funcion_modelo,
            nombre_libreria_origen=nombre_libreria_origen,
            params_modelo=params_modelo,
            params_fit=params_fit,
            nombre_X_fit='X_train',
            nombre_y_fit='y_train',
        )
        self.lista_modelos_seleccion_features.append(modelo)

        self.transformar_dataset(modelo, funcion_transformacion)

    def calcular_feature_extraction(
        self,
        nombre_funcion_modelo='PCA',
        nombre_libreria_origen='sklearn.decomposition',
        params_modelo=None,
    ):
        if params_modelo is None:
            params_modelo = dict()

        def f(modelo2, df):
            df2 = modelo2.transform(df)
            df2 = pd.DataFrame(df2)
            df2.columns = [f"{nombre_funcion_modelo}_{i}" for i in range(df2.shape[1])]
            return df2

        self.transformar_dataset_por_modelo(
            nombre_funcion_modelo=nombre_funcion_modelo,
            nombre_libreria_origen=nombre_libreria_origen,
            params_modelo=params_modelo,
            params_fit=dict(),
            funcion_transformacion=f,
        )

    def calcular_feature_selection(
        self,
        nombre_funcion_modelo,
        nombre_libreria_origen,
        params_modelo,
    ):
        if nombre_funcion_modelo == 'GARS':
            modelo_gars = self.seleccionar_features_por_gars(
                **params_modelo,
            )

        elif nombre_funcion_modelo == 'LassoCV':

            def f(modelo2, df):
                a = pd.Series(modelo2.coef_, index=modelo2.feature_names_in_)
                support = a.abs() > 0.0001
                df2 = df.loc[:, support]
                return df2

            self.transformar_dataset_por_modelo(
                nombre_funcion_modelo=nombre_funcion_modelo,
                nombre_libreria_origen=nombre_libreria_origen,
                params_modelo=params_modelo,
                params_fit=dict(),
                funcion_transformacion=f,
            )

        else:

            def f(modelo, df):
                mask = modelo.get_support()
                df2 = df.loc[:, mask]
                return df2

            self.transformar_dataset_por_modelo(
                nombre_funcion_modelo=nombre_funcion_modelo,
                nombre_libreria_origen=nombre_libreria_origen,
                params_modelo=params_modelo,
                params_fit=dict(),
                funcion_transformacion=f,
            )
