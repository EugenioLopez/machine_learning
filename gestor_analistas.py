# -*- coding: utf-8 -*-
# %%
"""Clase con la cual se gestiona una lista de analistas."""
import os
import pandas as pd
import numpy as np
import copy
from datetime import datetime

from parsers import Parser


# %%=============================================================================
# CLASE
# =============================================================================
class GestorAnalistas:
    """
    Clase que permite gestionar una lista de analistas.
    """

    # =========================================================================
    # CONSTRUCTOR
    # =========================================================================
    def __init__(self, lista_analistas):

        if not isinstance(lista_analistas, list):
            lista_analistas = [lista_analistas]

        self.lista_analistas = lista_analistas

    def predecir_individuos_por_filtro(
            self,
            d_filtros,
            usar_train=True,
            usar_test=True,
            usar_demo=True,
            calcular_proba=True,
    ):
        pass

    # Voting
    def predecir_por_voting(self,
                            d_filtros,
                            funcion_scorer,
                            parametros_scorer,
                            usar_train=True,
                            usar_test=True,
                            usar_demo=True,
                            corte=0.5,
                            voting='hard',
                            ):

        # Calculo las proba solo si es soft
        calcular_proba = (voting == 'soft')

        # Calculo las predicciones
        r = self.predecir_individuos_por_filtro(
            d_filtros,
            usar_train=usar_train,
            usar_test=usar_test,
            usar_demo=usar_demo,
            calcular_proba=calcular_proba,
        )
        res = {}

        if voting == 'hard':

            preds_train = r['y_predict_train']
            preds_test = r['y_predict_test']
            preds_demo = r['y_predict_demo']

            if usar_train:
                y_predict_train = (preds_train.mean(axis=1) > corte).astype(float)
                res.update(y_predict_train=y_predict_train)

            if usar_test:
                y_predict_test = (preds_test.mean(axis=1) > corte).astype(float)
                res.update(y_predict_test=y_predict_test)

            if usar_demo:
                y_predict_demo = (preds_demo.mean(axis=1) > corte).astype(float)
                res.update(y_predict_demo=y_predict_demo)

        elif voting == 'soft':

            preds_train = r['preds_proba_train_1']
            preds_test = r['preds_proba_test_1']
            preds_demo = r['preds_proba_demo_1']

            if usar_train:
                y_predict_train = (preds_train.mean(axis=1) > corte).astype(float)
                res.update(y_predict_train=y_predict_train)

            if usar_test:
                y_predict_test = (preds_test.mean(axis=1) > corte).astype(float)
                res.update(y_predict_test=y_predict_test)

            if usar_demo:
                y_predict_demo = (preds_demo.mean(axis=1) > corte).astype(float)
                res.update(y_predict_demo=y_predict_demo)

        else:
            raise TypeError('Error en la defincion del parametro voting.')

        # Calculo las metricas si la funcion_scorer no es None
        if funcion_scorer is not None:

            d = dict(funcion_scorer=funcion_scorer, parametros_scorer=parametros_scorer)

            if usar_train and self.y_train is not None:
                metr_tr = self.calcular_metrica2(y_true=self.y_train, y_predict=y_predict_train, **d)
                res.update(dict(metrica_train=metr_tr))

            if usar_test and self.y_test is not None:
                metr_te = self.calcular_metrica2(y_true=self.y_test, y_predict=y_predict_test, **d)
                res.update(dict(metrica_test=metr_te))

            if usar_demo and self.y_demo is not None:
                metr_de = self.calcular_metrica2(y_true=self.y_demo, y_predict=y_predict_demo, **d)
                res.update(dict(metrica_demo=metr_de))

        return res
