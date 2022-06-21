# -*- coding: utf-8 -*-
"""
Modulo que parametriza la ingenieria de modelos.
"""

# %%===========================================================================
# # IMPORTO
# =============================================================================
# GENERALES
import os
import copy
import numpy as np

# import pandas as pd

# SKLEARN
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split

# from sklearn.feature_selection import SelectFromModel
# from sklearn.feature_selection import RFECV
# from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold

# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit

from sklearn.utils import shuffle
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
# from scikeras.wrappers import KerasClassifier

# from redes_neuronales import perceptron_multicapa
from data_manager import train_test_split_algo_genetico

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# %%===========================================================================
# # Scalers
# =============================================================================
def scalers_base():
    scalers = dict()
    scalers["StandardScaler"] = StandardScaler()
    scalers["MinMaxScaler"] = MinMaxScaler()
    scalers["RobustScaler"] = RobustScaler()
    scalers["UniformQuantileTransformer"] = QuantileTransformer(
        output_distribution="uniform"
    )
    scalers["NormalQuantileTransformer"] = QuantileTransformer(
        output_distribution="normal"
    )
    scalers["PowerTransformer"] = PowerTransformer()

    return scalers


# %%===========================================================================
# Estimadores
# =============================================================================
# Generico
def estimador_base(funcion_estimador, params_estimador, param_grid, lista_param_grid):

    # Creo el diccionario
    estimador = dict()

    # Armo el estimador
    if type(funcion_estimador) == str or type(funcion_estimador) is np.str_:
        exec(f"""estimador['estimator'] = {funcion_estimador}(**params_estimador)""")

    else:
        estimador["estimator"] = funcion_estimador(**params_estimador)

    # Storeo la info
    estimador["funcion_estimador"] = funcion_estimador
    estimador["param_grid"] = param_grid
    estimador["lista_param_grid"] = lista_param_grid

    return estimador


# dict_estimador = generar_estimador(funcion_estimador='KNeighborsClassifier')
def generar_estimador(
    funcion_estimador,
    dimension_features=None,
    params_estimador=None,
    param_grid=None,
    lista_param_grid=None,
):

    # KNeighbors
    if (
        funcion_estimador == KNeighborsClassifier
        or funcion_estimador == "KNeighborsClassifier"
    ):

        if not params_estimador:
            params_estimador = dict(n_jobs=-1, n_neighbors=3)

        if not param_grid:
            param_grid = {
                "n_neighbors": list(range(2, 30)),
                "weights": ["uniform", "distance"],
            }

        if not lista_param_grid:
            lista_param_grid = [
                {"n_neighbors": list(range(2, 30)), "weights": ["uniform", "distance"]}
            ]

    # LOGISTIC
    elif (
        funcion_estimador == LogisticRegression
        or funcion_estimador == "LogisticRegression"
    ):

        if not params_estimador:
            params_estimador = dict(
                C=2.0, solver="liblinear", random_state=0, warm_start=True
            )

        if not param_grid:
            param_grid = {
                "C": [1, 1.5, 2, 5, 10, 50, 100, 500, 1000],
                "penalty": ["l1", "l2"],
            }

        if not lista_param_grid:
            lista_param_grid = [{"C": [2, 5, 10, 50, 100]}, {"penalty": ["l1", "l2"]}]

    # LOGISTIC CV
    elif (
        funcion_estimador == LogisticRegressionCV
        or funcion_estimador == "LogisticRegressionCV"
    ):

        if not params_estimador:
            params_estimador = dict(
                Cs=2, solver="liblinear", random_state=0, cv=StratifiedKFold()
            )

        if not param_grid:
            param_grid = {
                "Cs": [1, 1.5, 2, 5, 10, 50, 100, 500, 1000],
                "penalty": ["l1", "l2"],
            }

        if not lista_param_grid:
            lista_param_grid = [
                {"Cs": [1, 1.5, 2, 5, 10, 50, 100, 500, 1000], "penalty": ["l1", "l2"]}
            ]

    # Random Forest
    elif (
        funcion_estimador == RandomForestClassifier
        or funcion_estimador == "RandomForestClassifier"
    ):

        if not params_estimador:
            params_estimador = dict(
                n_jobs=-1,
                oob_score=True,
                max_depth=3,
                criterion="gini",
                random_state=0,
                n_estimators=250,
            )

        if not param_grid:
            param_grid = {
                "n_estimators": [100, 250],
                "criterion": ["gini", "entropy"],
                "max_depth": [2, 3, 4, 5, 10],
                "oob_score": [True, False],
                "class_weight": ["balanced", "balanced_subsample"],
                "random_state": [0],
                "max_samples": [0.1, 0.5, 0.99],
            }

        if not lista_param_grid:
            lista_param_grid = [
                {"criterion": ["gini", "entropy"], "max_depth": [2, 3, 4, 5, 10]},
                {
                    "oob_score": [True, False],
                    "class_weight": ["balanced", "balanced_subsample"],
                    "random_state": [0],
                    "max_samples": [0.1, 0.5, 0.99],
                },
                {"n_estimators": [100, 250, 400]},
            ]

    # XGB
    elif funcion_estimador == XGBClassifier or funcion_estimador == "XGBClassifier":

        if not params_estimador:
            params_estimador = dict(
                n_jobs=-1,
                max_depth=2,
                use_label_encoder=False,
                random_state=0,
                eval_metric="mlogloss",
                n_estimators=250,
            )

        if not param_grid:
            param_grid = dict(
                max_depth=[2, 3, 4, 5, None], subsample=[0.5], use_label_encoder=[False]
            )

        if not lista_param_grid:
            lista_param_grid = [
                {
                    "gamma": [0.01, 0.5, 1.0, 2.0, 5.0, 10.0],
                    "max_depth": [2, 3, 4, 5, 10, 20],
                },
                {
                    "colsample_bytree": [0.1, 0.5, 0.9],
                    "colsample_bylevel": [0.1, 0.5, 0.9],
                    "colsample_bynode": [0.1, 0.5, 0.9],
                },
                {
                    "min_child_weight": [0.1, 0.25, 0.5, 1.0],
                    "max_delta_step": [0.0, 0.5, 1.0, 10.0],
                },
                {"reg_alpha": [0.0, 0.5, 1.0, 5.0], "reg_lambda": [0.0, 0.5, 1.0, 5.0]},
                {
                    "learning_rate": [0.01, 0.3, 0.1, 0.5, 0.75, 1.0],
                    "n_estimators": [100, 300, 500],
                },
            ]

    # Red Neuronal
    # elif funcion_estimador == KerasClassifier or funcion_estimador == "KerasClassifier": #TODO
    #
    #     if not params_estimador:
    #         params_estimador = dict(
    #             model=perceptron_multicapa,
    #             dimension_features=dimension_features,
    #             cantidad_capas_ocultas=10,
    #             neuronas_capas_ocultas=int(dimension_features * 0.8),
    #             neuronas_capa_final=1,
    #             prob_no_activacion=0.2,
    #             activacion_capas_ocultas="relu",
    #             activacion_capa_final="sigmoid",
    #             optimizador="adam",
    #             metricas=["accuracy"],
    #         )
    #
    #     if not param_grid:
    #         param_grid = {
    #             "cantidad_capas_ocultas": np.array([50]),
    #             "epochs": np.array([100]),
    #             "batch_size": np.array([1, 10, 100]),
    #         }
    #
    #     if not lista_param_grid:
    #         lista_param_grid = [
    #             {
    #                 "cantidad_capas_ocultas": np.array([50]),
    #                 "epochs": np.array([100]),
    #                 "batch_size": np.array([1, 10, 100]),
    #             }
    #         ]

    # SVC
    elif funcion_estimador == SVC or funcion_estimador == "SVC":

        if not params_estimador:
            params_estimador = dict(probability=True, C=1.0, random_state=0)

        if not param_grid:
            param_grid = {
                "C": [0.5, 1, 10, 100],
                "gamma": ["scale", 1, 0.1, 0.01, 0.001, 0.0001],
            }

        if not lista_param_grid:
            lista_param_grid = [
                {
                    "C": [0.5, 1, 10, 100],
                    "gamma": ["scale", 1, 0.1, 0.01, 0.001, 0.0001],
                }
            ]

    # GradientBoostingClassifier
    elif (
        funcion_estimador == GradientBoostingClassifier
        or funcion_estimador == "GradientBoostingClassifier"
    ):

        if not params_estimador:
            params_estimador = dict(random_state=0)

        if not param_grid:
            param_grid = dict(loss=["deviance", "exponential"], n_estimators=[100, 250])

        if not lista_param_grid:
            lista_param_grid = [
                dict(loss=["deviance", "exponential"], n_estimators=[100, 250])
            ]

    # Tree
    elif (
        funcion_estimador == DecisionTreeClassifier
        or funcion_estimador == "DecisionTreeClassifier"
    ):

        if not params_estimador:
            params_estimador = dict(random_state=0)

        if not param_grid:
            param_grid = {}

        if not lista_param_grid:
            lista_param_grid = [{}]

    # Bagging Tree
    elif (
        funcion_estimador == BaggingClassifier
        or funcion_estimador == "BaggingClassifier"
    ):

        if not params_estimador:
            params_estimador = dict(
                n_estimators=250, n_jobs=-1, bootstrap_features=True, oob_score=True
            )

        if not param_grid:
            dict_base = generar_estimador(
                DecisionTreeClassifier,
                dimension_features,
                params_estimador=None,
                param_grid=None,
                lista_param_grid=None,
            )

            param_grid = dict(n_estimators=[250])
            params_temp = {
                f"base_estimator__{k}": v for k, v in dict_base["param_grid"].items()
            }
            param_grid.update(params_temp)

        if not lista_param_grid:
            dict_base = generar_estimador(
                DecisionTreeClassifier,
                dimension_features,
                params_estimador=None,
                param_grid=None,
                lista_param_grid=None,
            )
            param_grid = dict(n_estimators=[250])
            params_temp = {
                f"base_estimator__{k}": v for k, v in dict_base["param_grid"].items()
            }
            param_grid.update(params_temp)
            lista_param_grid = [param_grid]

    # Castpool
    elif (
        funcion_estimador == CatBoostClassifier
        or funcion_estimador == "CatBoostClassifier"
    ):

        if not params_estimador:
            params_estimador = dict(
                iterations=200,
                depth=2,
                learning_rate=1,
                loss_function="Logloss",
                verbose=False,
                random_state=0,
            )

        if not param_grid:
            param_grid = dict(depth=[2, 3, 5, 10, 20])

        if not lista_param_grid:
            lista_param_grid = [dict(depth=[2, 3, 5, 10, 20])]

    # lightgbm
    elif funcion_estimador == LGBMClassifier or funcion_estimador == "LGBMClassifier":

        if not params_estimador:
            params_estimador = dict(learning_rate=0.1, max_bin=100, random_state=0)

        if not param_grid:
            param_grid = dict(num_leaves=[2, 4, 8, 16, 32, 64, 100, 1000])

        if not lista_param_grid:
            lista_param_grid = [dict(num_leaves=[2, 4, 8, 16, 32, 64, 100, 1000])]

    else:

        if not params_estimador:
            params_estimador = dict()

        if not param_grid:
            param_grid = dict()

        if not lista_param_grid:
            lista_param_grid = []

    return estimador_base(
        funcion_estimador,
        params_estimador=params_estimador,
        param_grid=param_grid,
        lista_param_grid=lista_param_grid,
    )


def aplicar_bagging(dict_base, param_grid):

    params_estimador = dict(
        n_estimators=250, n_jobs=-1, bootstrap_features=True, oob_score=True
    )

    # Armo el diccionario de param_grid base
    params_temp = {
        f"base_estimator__{k}": v for k, v in dict_base["param_grid"].items()
    }
    param_grid.update(params_temp)
    lista_param_grid = [dict_base["param_grid"]]

    dict_base2 = estimador_base(
        BaggingClassifier,
        params_estimador=params_estimador,
        param_grid=dict_base["param_grid"],
        lista_param_grid=lista_param_grid,
    )

    return dict_base2


# =============================================================================
# CREAR MODELOS
# =============================================================================
def crear_modelos_cv(f_estimador, param_fijos=None, param_grid=dict(), barajado=False):

    if param_fijos is None:
        param_fijos = {
            "max_depth": list(range(1, 11, 2)),
            "n_estimators": [100],
            "min_samples_split": list(np.linspace(0, 0.5, 3)),
            "min_samples_leaf": list(np.linspace(0, 0.5, 3)),
            "oob_score": [True, False],
            "n_jobs": [-1],
        }

    # Creo el vector de combinaciones
    if barajado:
        comb_params = shuffle(ParameterGrid([param_fijos]))
    else:
        comb_params = ParameterGrid([param_fijos])

    # creo la lista de estimadores
    rfs_lista = list()
    for el in list(comb_params):
        rfs = dict()
        rfs["estimator"] = f_estimador.set_params(**el)
        rfs["param_grid"] = dict()
        rfs_lista.append(copy.deepcopy(rfs))

    return rfs_lista


# %%===========================================================================
# FUNCION DE DICCIONARIOS PARA CALIBRAR EL AJUSTE POR ALGORITMO GENETICO
# =============================================================================
def generar_parametros_algoritmo_genetico(
    X_analisis,
    lista_estimadores,
    agregar_hiperparametros=True,
    agregar_total_hiperparametros=True,
    filtrar_scaler=False,
    filtrar_columnas=True,
    probabilidad_ocurrencia_columna=0.5,
    filtrar_filas=True,
    probabilidad_ocurrencia_fila=0.5,
    dd_parametros_adicionales=None,
):

    # Scalers
    d_scalers = dict()
    if filtrar_scaler:
        d_scalers.update(
            dict(scaler__scaler=dict(type=list, lista=list(scalers_base().values())))
        )

    # Columnas
    d_columnas = dict()
    if filtrar_columnas:

        p = probabilidad_ocurrencia_columna
        d_columnas.update(
            {
                f"columnas__numeroColumna_{col}": dict(
                    type=list, lista=[True, False], p=[p, 1.0 - p]
                )
                for col in range(int(X_analisis.shape[1]))
            }
        )

    # Filas
    d_filas = dict()
    if filtrar_filas:

        p = probabilidad_ocurrencia_fila
        d_filas.update(
            {
                f"filas__{fila}": dict(type=list, lista=[True, False], p=[p, 1.0 - p])
                for fila in range(int(X_analisis.shape[0] + 1))
            }
        )

    # Estimador
    d_estimador = dict()
    d_estimador.update(
        dict(estimador__funcion_estimador=dict(type=list, lista=lista_estimadores))
    )

    # Parametros
    d_parametros = dict()
    d_parametros.update(
        dict(params__funcion_split_train_test=dict(type=list, lista=[train_test_split]))
    )
    d_parametros.update(
        dict(
            params__parametros_split_train_test=dict(
                type=list, lista=[dict(test_size=0.3, shuffle=True)]
            )
        )
    )
    d_parametros.update(
        dict(params__funcion_split_test_demo=dict(type=list, lista=[train_test_split]))
    )
    d_parametros.update(
        dict(
            params__parametros_split_test_demo=dict(
                type=list, lista=[dict(test_size=0.5, shuffle=True)]
            )
        )
    )
    d_parametros.update(dict(params__parametros_scaler=dict(type=list, lista=[dict()])))
    d_parametros.update(dict(params__ajustar=dict(type=list, lista=[True])))
    d_parametros.update(dict(params__tipo_cv=dict(type=list, lista=["grid"])))
    d_parametros.update(
        dict(
            params__parametros_grid=dict(
                type=list,
                lista=[
                    dict(
                        n_jobs=-1,
                        verbose=0,
                        cv=ShuffleSplit(n_splits=30, train_size=0.5, test_size=0.25),
                    )
                ],
            )
        )
    )  # LeaveOneOut() cv=StratifiedKFold(n_splits=5, shuffle=True
    d_parametros.update(dict(params__parametros_fit=dict(type=list, lista=[dict()])))
    d_parametros.update(
        dict(params__funcion_scorer=dict(type=list, lista=["accuracy_score"]))
    )
    d_parametros.update(dict(params__parametros_scorer=dict(type=list, lista=[dict()])))
    d_parametros.update(dict(params__verbose=dict(type=list, lista=[0])))

    # Completo los parametros si no estan en el individuo
    # Hiperparametros
    d_hiperparametros = dict()

    if agregar_hiperparametros:

        if LGBMClassifier in lista_estimadores or "LGBMClassifier" in lista_estimadores:
            d_hiperparametros.update(
                dict(
                    hiperparametros__LGBMClassifier_num_leaves=dict(
                        type=int, low=2, high=500
                    )
                )
            )
            d_hiperparametros.update(
                dict(
                    hiperparametros__LGBMClassifier_learning_rate=dict(
                        type=float, low=0.01, high=0.999
                    )
                )
            )
            d_hiperparametros.update(
                dict(
                    hiperparametros__LGBMClassifier_n_estimators=dict(
                        type=int, low=50, high=500
                    )
                )
            )
            d_hiperparametros.update(
                dict(
                    hiperparametros__LGBMClassifier_boosting_type=dict(
                        type=list, lista=["gbdt", "dart"]
                    )
                )
            )  # lista=['gbdt', 'dart']
            d_hiperparametros.update(
                dict(
                    hiperparametros__LGBMClassifier_objective=dict(
                        type=list, lista=["binary"]
                    )
                )
            )
            d_hiperparametros.update(
                dict(
                    hiperparametros__LGBMClassifier_verbose=dict(
                        type=list, lista=[-100]
                    )
                )
            )
            d_hiperparametros.update(
                dict(
                    hiperparametros__LGBMClassifier_max_depth=dict(
                        type=int, low=2, high=15
                    )
                )
            )

            if agregar_total_hiperparametros:
                # d_hiperparametros.update(dict(hiperparametros__LGBMClassifier_min_gain_to_split = dict(type=float, low=0., high=1.)))
                # d_hiperparametros.update(dict(hiperparametros__LGBMClassifier_min_data_in_leaf = dict(type=int, low=20, high=200)))
                # d_hiperparametros.update(dict(hiperparametros__LGBMClassifier_feature_fraction = dict(type=list, lista=[0.25, 0.5, 0.75, 1.])))
                d_hiperparametros.update(
                    dict(
                        hiperparametros__LGBMClassifier_bagging_fraction=dict(
                            type=list, lista=[0.25, 0.5, 0.75, 1.0]
                        )
                    )
                )
                # d_hiperparametros.update(dict(hiperparametros__LGBMClassifier_bagging_freq = dict(type=int, low=10, high=50)))
                d_hiperparametros.update(
                    dict(
                        hiperparametros__LGBMClassifier_min_child_weight=dict(
                            type=int, low=10, high=100
                        )
                    )
                )
                d_hiperparametros.update(
                    dict(
                        hiperparametros__LGBMClassifier_subsample=dict(
                            type=list, lista=[0.0, 0.25, 0.5, 0.75, 1.0]
                        )
                    )
                )
                d_hiperparametros.update(
                    dict(
                        hiperparametros__LGBMClassifier_reg_alpha=dict(
                            type=list, lista=[0.0, 0.25, 0.5, 0.75, 1.0]
                        )
                    )
                )
                d_hiperparametros.update(
                    dict(
                        hiperparametros__LGBMClassifier_reg_lambda=dict(
                            type=list, lista=[0.0, 0.25, 0.5, 0.75, 1.0]
                        )
                    )
                )
                # d_hiperparametros.update(dict(hiperparametros__LGBMClassifier_colsample_bytree = dict(type=list, lista=[0., 0.25, 0.5, 0.75, 1.])))

        if (
            LogisticRegression in lista_estimadores
            or "LogisticRegression" in lista_estimadores
        ):
            d_hiperparametros.update(
                dict(
                    hiperparametros__LogisticRegression_C=dict(
                        type=list,
                        lista=[
                            0.001,
                            0.01,
                            0.1,
                            0.5,
                            1.0,
                            1.5,
                            2.0,
                            5.0,
                            10.0,
                            20.0,
                            50.0,
                            100.0,
                            200.0,
                            500.0,
                            1000.0,
                        ],
                    )
                )
            )
            d_hiperparametros.update(
                dict(
                    hiperparametros__LogisticRegression_penalty=dict(
                        type=list, lista=["l1", "l2"]
                    )
                )
            )

        if XGBClassifier in lista_estimadores or "XGBClassifier" in lista_estimadores:
            d_hiperparametros.update(
                dict(
                    hiperparametros__XGBClassifier_learning_rate=dict(
                        type=float, low=0.01, high=1.0
                    )
                )
            )
            d_hiperparametros.update(
                dict(
                    hiperparametros__XGBClassifier_min_child_weight=dict(
                        type=float, low=0.01, high=10.0
                    )
                )
            )
            d_hiperparametros.update(
                dict(
                    hiperparametros__XGBClassifier_max_depth=dict(
                        type=int, low=1, high=10
                    )
                )
            )
            d_hiperparametros.update(
                dict(
                    hiperparametros__XGBClassifier_n_estimators=dict(
                        type=int, low=100, high=1500
                    )
                )
            )

            if agregar_total_hiperparametros:
                d_hiperparametros.update(
                    dict(
                        hiperparametros__XGBClassifier_gamma=dict(
                            type=float, low=0.01, high=2.0
                        )
                    )
                )
                d_hiperparametros.update(
                    dict(
                        hiperparametros__XGBClassifier_max_delta_step=dict(
                            type=float, low=0.0, high=10.0
                        )
                    )
                )
                d_hiperparametros.update(
                    dict(
                        hiperparametros__XGBClassifier_subsample=dict(
                            type=float, low=0.1, high=1.0
                        )
                    )
                )
                d_hiperparametros.update(
                    dict(
                        hiperparametros__XGBClassifier_colsample_bytree=dict(
                            type=float, low=0.1, high=1.0
                        )
                    )
                )
                d_hiperparametros.update(
                    dict(
                        hiperparametros__XGBClassifier_colsample_bylevel=dict(
                            type=float, low=0.01, high=1.0
                        )
                    )
                )
                d_hiperparametros.update(
                    dict(
                        hiperparametros__XGBClassifier_colsample_bynode=dict(
                            type=float, low=0.01, high=1.0
                        )
                    )
                )
                d_hiperparametros.update(
                    dict(
                        hiperparametros__XGBClassifier_scale_pos_weight=dict(
                            type=float, low=0.01, high=2.0
                        )
                    )
                )

        if (
            GradientBoostingClassifier in lista_estimadores
            or "GradientBoostingClassifier" in lista_estimadores
        ):
            d_hiperparametros.update(
                dict(
                    hiperparametros__GradientBoostingClassifier_learning_rate=dict(
                        type=float, low=0.1, high=1.0
                    )
                )
            )
            d_hiperparametros.update(
                dict(
                    hiperparametros__GradientBoostingClassifier_max_depth=dict(
                        type=int, low=1, high=10
                    )
                )
            )
            d_hiperparametros.update(
                dict(
                    hiperparametros__GradientBoostingClassifier_n_estimators=dict(
                        type=int, low=50, high=500
                    )
                )
            )

            if agregar_total_hiperparametros:
                d_hiperparametros.update(
                    dict(
                        hiperparametros__GradientBoostingClassifier_subsample=dict(
                            type=list, lista=[0.25, 0.5, 0.75, 1.0]
                        )
                    )
                )

        if CatBoostClassifier in lista_estimadores:
            d_hiperparametros.update(
                dict(
                    hiperparametros__CatBoostClassifier_learning_rate=dict(
                        type=float, low=0.01, high=1.0
                    )
                )
            )
            d_hiperparametros.update(
                dict(
                    hiperparametros__CatBoostClassifier_max_depth=dict(
                        type=int, low=1, high=10
                    )
                )
            )
            d_hiperparametros.update(
                dict(
                    hiperparametros__CatBoostClassifier_iterations=dict(
                        type=int, low=100, high=1500
                    )
                )
            )

            if agregar_total_hiperparametros:
                d_hiperparametros.update(
                    dict(
                        hiperparametros__CatBoostClassifier_subsample=dict(
                            type=list, lista=[0.0, 0.25, 0.5, 0.75, 1.0]
                        )
                    )
                )
                d_hiperparametros.update(
                    dict(
                        hiperparametros__CatBoostClassifier_scale_pos_weight=dict(
                            type=float, low=0.01, high=2.0
                        )
                    )
                )

    # unifico los diccionarios
    d_total = dict()
    d_total.update(d_columnas)
    d_total.update(d_filas)
    d_total.update(d_scalers)
    d_total.update(d_estimador)
    d_total.update(d_parametros)
    d_total.update(d_hiperparametros)

    # Actualizo el diccionario con los parametros finales
    if type(dd_parametros_adicionales) == dict:
        d_total.update(dd_parametros_adicionales)

    return d_total


# =============================================================================
# OTROS MODELOS DESARROLLADOS
# =============================================================================

# =============================================================================
# Voting
# # =============================================================================
# def voting_calibrado(dict_estimadores=None, param_grid=None):

#     # Tomo los estimadores
#     if dict_estimadores is None:
#         dict_estimadores = dict()
#         dict_estimadores['xgb'] = xgb_base()
#         dict_estimadores['randomForest'] = randomForest_base()
#         dict_estimadores['bagging_tree'] = aplicar_bagging(tree_base())

#     dict_estimadores = {k: aplicar_calibration(v)
#                         for k, v in dict_estimadores.items()}

#     if param_grid is None:
#         param_grid = {'method': ['sigmoid', 'isotonic']}

#     voting = voting_base(dict_estimadores=dict_estimadores,
#                          param_grid=param_grid)

#     return voting

# %%===========================================================================
# Aplico calibracion
# =============================================================================
def aplicar_calibration(dict_base, param_grid=None):

    # Armo el estimador
    dict_base2 = dict_base.copy()
    dict_base2["estimator"] = CalibratedClassifierCV(
        dict_base["estimator"], method="sigmoid"
    )

    # Armo el diccionario de param_grid base
    if param_grid is None:
        dict_base2["param_grid"] = dict(method=["sigmoid", "isotonic"])
    else:
        dict_base2["param_grid"] = param_grid

    return dict_base2


# %%=============================================================================
# # Perceptron multicapa
# =============================================================================
# def perceptron_base(  # TODO
#     dimension_features,
#     cantidad_capas_ocultas,
#     neuronas_capas_ocultas,
#     neuronas_capa_final,
#     prob_no_activacion,
#     activacion_capas_ocultas,
#     activacion_capa_final,
#     optimizador,
#     metricas,
#     param_grid=None,
# ):
#
#     # Creo el diccionario
#     perc = dict()
#     model = KerasClassifier(
#         model=perceptron_multicapa,
#         dimension_features=dimension_features,
#         cantidad_capas_ocultas=cantidad_capas_ocultas,
#         neuronas_capas_ocultas=neuronas_capas_ocultas,
#         neuronas_capa_final=neuronas_capa_final,
#         prob_no_activacion=prob_no_activacion,
#         activacion_capas_ocultas=activacion_capas_ocultas,
#         activacion_capa_final=activacion_capa_final,
#         optimizador=optimizador,
#         metricas=metricas,
#     )
#
#     perc["estimador"] = model
#
#     # Define the grid search parameters
#     if param_grid is None:
#         perc["param_grid"] = dict(
#             epochs=list(range(50, 101, 10)),
#             cantidad_capas_ocultas=list(range(1, 50)),
#             neuronas_capas_ocultas=list(range(20, 35)),
#         )
#     else:
#         perc["param_grid"] = param_grid
#
#     return perc
