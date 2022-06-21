# -*- coding: utf-8 -*-
"""
Módulo en el que se trabaja con redes neuranales.
"""
import numpy as np
import pandas as pd
import os

from scikeras.wrappers import KerasRegressor
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# tf.compat.v1.disable_eager_execution()


# %=============================================================================
# # Function to create model, required for KerasClassifier
# =============================================================================
# def perceptron_multicapa(dimension_features=162,
#                          cantidad_capas_ocultas=1,
#                          neuronas_capas_ocultas=8,
#                          neuronas_capa_final=1,
#                          prob_no_activacion=0.2,
#                          activacion_capas_ocultas='relu',
#                          activacion_capa_final='softmax',
#                          loss='binary_crossentropy',
#                          optimizador='adam',
#                          metricas=['accuracy']):
#     """
#     Funcion con la que se crea una red neuronal con formato de perceptron,
#     el cual se caracteriza por una capa inicial, capas intermedias de igual
#     cantidad de neuronas (que pueden tener la misma o distintas funciones
#     de activacion) y una capa final.
#
#     Paramettros
#     -----------
#
#     dimension_features: int. Opcional. Default: 8
#         Parametro que indica la cantidad de features que seran inputs de
#         la red neuronal.
#
#     cantidad_capas_ocultas: int. Opcional. Default: 1
#         Parametro que indica la cantidad de capas ocultas que va a tener
#         el perceptron.
#
#     neuronas_capas_ocultas: int, float o list. Opcional. Default: 8
#         Parametro que indica la cantidad de neuronas que van a tener las capas
#         intermedias del perceptron. En caso de que sea una lista, la
#         dimension debe conincidir con la cantidad_capas_ocultas.
#
#     neuronas_capa_final: int. Opcional. Default: 1
#         Parametro que indica la cantidad de neuronas que van a tener la capa
#         final del perceptron.
#
#     prob_no_activacion: float. Opcional. Default: 0.2
#         Parametro que establece la probabilidad de no activacion de las neuro-
#         nas de la capa inicial e intermedias, que servira para evitar
#         sobreajuste.
#
#     activacion_capas_ocultas: string o list. Opcional. Default 'relu'.
#         Parametro que indica la función que van a tener las capas ocultas.
#         En caso de que sea una lista, la dimension debe conincidir con la
#         cantidad_capas_ocultas.
#
#     activacion_final: string. Opcional. Default 'softmax'.
#         Parametro que indica la función que van a tener la capa final.
#
#     optimizador: string. Opcional. Default 'adam'.
#         Parametro que establecie cual va a ser el optimizador del perceptron.
#
#     metricas : list. Opcional. Default: ['accuracy'].
#         Lista con las metricas a utilizarse en la optimizacion del perceptron.
#
#     Retorno
#     -------
#     model: tensorflow.python.keras.engine.sequential.Sequential
#         Red Neuronal con el formato de perceptron.
#
#     """
#     if type(neuronas_capas_ocultas) is int or type(neuronas_capas_ocultas) is float:
#         neuronas_capas_ocultas = [
#                                      neuronas_capas_ocultas] * cantidad_capas_ocultas
#
#     if type(activacion_capas_ocultas) is str:
#         activacion_capas_ocultas = [
#                                        activacion_capas_ocultas] * cantidad_capas_ocultas
#
#     if len(neuronas_capas_ocultas) != len(activacion_capas_ocultas):
#         a = ''.join(['Error en las dimensiones de neuronas_capas_ocultas ó ',
#                      'activacion_capas_ocultas'])
#         raise ValueError(a)
#
#     # Creo el modelo
#     model = Sequential()
#
#     # Agrego las capas intermedias
#     elementos = list(
#         enumerate(zip(neuronas_capas_ocultas, activacion_capas_ocultas)))
#
#     for num, (neuronas, activacion_capa) in elementos:
#
#         # Agrego la capa inicial
#         if num == 0:
#             model.add(Dense(neuronas,
#                             input_dim=dimension_features,
#                             kernel_initializer='uniform',
#                             activation=activacion_capa))
#
#         else:
#             model.add(Dense(neuronas,
#                             kernel_initializer='uniform',
#                             activation=activacion_capa))
#
#         model.add(Dropout(prob_no_activacion))
#
#     # Agrego la capa final
#     model.add(Dense(neuronas_capa_final,
#                     activation=activacion_capa_final))
#
#     # Compile model
#     model.compile(loss=loss,
#                   optimizer=optimizador,
#                   metrics=metricas)
#
#     return model


def funcion_crea_nn(dimension_features=161,
                    cantidad_capas_ocultas=1,
                    cantidad_neuronas_por_capa_oculta=8,
                    neuronas_capa_final=1,
                    prob_no_activacion=0.2,
                    f_activacion_capas_ocultas='relu',
                    f_activacion_capa_final='softmax',
                    optimizador='adam',
                    metricas=['accuracy']):

    def perceptron_multicapa(dimension_features,
                             cantidad_capas_ocultas,
                             cantidad_neuronas_por_capa_oculta,
                             neuronas_capa_final,
                             prob_no_activacion,
                             f_activacion_capas_ocultas,
                             f_activacion_capa_final,
                             optimizador,
                             metricas):

        if type(cantidad_neuronas_por_capa_oculta) is int or type(cantidad_neuronas_por_capa_oculta) is float:
            temp = [cantidad_neuronas_por_capa_oculta] * cantidad_capas_ocultas
            cantidad_neuronas_por_capa_oculta = temp

        if type(f_activacion_capas_ocultas) is str:
            temp = [f_activacion_capas_ocultas] * cantidad_capas_ocultas
            f_activacion_capas_ocultas = temp

        if len(cantidad_neuronas_por_capa_oculta) != len(cantidad_neuronas_por_capa_oculta):
            a = ''.join(['Error en las dimensiones de neuronas_capas_ocultas ó ',
                         'f_activacion_capas_ocultas'])
            raise ValueError(a)

        # Creo el modelo
        model = Sequential()

        # Agrego las capas intermedias
        elementos = list(
            enumerate(zip(cantidad_neuronas_por_capa_oculta, f_activacion_capas_ocultas)))

        for num, (neuronas, activacion_capa) in elementos:

            # Agrego la capa inicial
            if num == 0:
                model.add(
                    Dense(
                        neuronas,
                        input_dim=dimension_features,
                        kernel_initializer='uniform',
                        activation=activacion_capa
                    )
                )

            # Agrego las demás capas
            else:
                model.add(
                    Dense(
                        neuronas,
                        kernel_initializer='uniform',
                        activation=activacion_capa
                    )
                )

            # Agrego el dropout si corresponde
            if prob_no_activacion > 0:
                model.add(Dropout(prob_no_activacion))

        # Agrego la capa final
        model.add(
            Dense(
                neuronas_capa_final,
                activation=f_activacion_capa_final
            )
        )

        # Compile model
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizador,
                      metrics=metricas)

        return model

    return perceptron_multicapa


# %%===========================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':

    # =========================================================================
    # Genero los datos
    # =========================================================================
    # fix random seed for reproducibility
    np.random.seed(7)

    # Load data
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin',
             'test', 'mass', 'pedi', 'age', 'class']
    dataset = pd.read_csv(url, names=names)

    # split into input (X) and output (Y) variables
    X = dataset.iloc[:, 0:8]
    Y = dataset.iloc[:, 8]

# %%=============================================================================
# Creo el modelo
# =============================================================================
if __name__ == '__main__':

    model = funcion_crea_nn()

    estimator = KerasRegressor(
        model=model,
        dimension_features=self.X_train.shape[1],
        cantidad_capas_ocultas=5,
        cantidad_neuronas_por_capa_oculta=int(self.X_train.shape[1] * 1.2),
        neuronas_capa_final=1,
        prob_no_activacion=0.3,
        f_activacion_capas_ocultas='relu',
        f_activacion_capa_final='sigmoid',
        optimizador='adam',
        metricas=['accuracy']
    )

    X = self.X_train
    y = self.y_train
    X_valid = self.X_test
    y_valid = self.y_test

    estimator = estimator.fit(
        X,
        y,
        epochs=500,
        batch_size=1,
        validation_data=(X_valid, y_valid),
    )


    pr = estimator.predict(X)
    pr_test = estimator.predict(self.X_test)
    pr_test2 = pr_test.round()

    m = self.calcular_metrica2(
        y_true=self.y_test,
        y_predict=pr_test2,
        funcion_scorer='accuracy_score',
        parametros_scorer=None,
    )




def eval_metric(model, history, metric_name):
    '''
    Function to evaluate a trained model on a chosen metric.
    Training and validation metric are plotted in a
    line chart for each epoch.

    Parameters:
        history : model training history
        metric_name : loss or accuracy
    Output:
        line chart with epochs of x-axis and metric on
        y-axis
    '''
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]
    e = range(1, NB_START_EPOCHS + 1)
    plt.plot(e, metric, 'bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
    plt.xlabel('Epoch number')
    plt.ylabel(metric_name)
    plt.title('Comparing training and validation ' + metric_name + ' for ' + model.name)
    plt.legend()
    plt.show()

def optimal_epoch(model_hist):
    '''
    Function to return the epoch number where the validation loss is
    at its minimum

    Parameters:
        model_hist : training history of model
    Output:
        epoch number with minimum validation loss
    '''
    min_epoch = np.argmin(model_hist.history['val_loss']) + 1
    print("Minimum validation loss reached in epoch {}".format(min_epoch))
    return min_epoch

eval_metric(model=model, history=estimator, metric_name='accuracy_score')