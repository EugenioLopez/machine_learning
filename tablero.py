# -*- coding: utf-8 -*-
# %%
"""Clase con la cual se parametriza un tablero de analistas"""
import os
import copy
import pandas as pd

from parsers import Parser
from analista import Analista


# %%=============================================================================
# CLASE
# =============================================================================
class Tablero:
    """
    Clase que simula a un tablero de analistas de machine learning.
    """

    # =========================================================================
    # CONSTRUCTOR
    # =========================================================================
    def __init__(self, nombre, carpeta):

        # Copio los valores
        self.nombre = nombre
        self.carpeta = carpeta
        self.parser = Parser(nombre, carpeta)

        self.tablero = None

    # Funciones de guardado
    def guardar(self, prefix="", suffix=""):

        # armo el nombre del archivo
        nombre_archivo = f"{prefix}{self.nombre}{suffix}"

        # Cargo el diccionario, borro el parser
        d_data = copy.copy(self.__dict__)
        del d_data["parser"]

        # Guardo
        self.parser.pickleo(d_data, nombre_archivo)

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

    def cargar_versiones(self, nombre_analistas, ind=None):

        # Completo la carpeta
        carpeta = self.carpeta

        # Tomo los nombres de los archivos dentro de la carpeta con nombre similar
        l_archivos = [a for a in os.listdir(carpeta) if nombre_analistas in a]

        # Tomo la lista de índices de los archivos que voy a cargar
        if isinstance(ind, int):
            l_archivos = [l_archivos[ind]]

        elif isinstance(ind, list):
            l_archivos = [l_archivos[i] for i in ind]

        elif ind is None:
            l_archivos = l_archivos

        else:
            raise TypeError('Error en el parametro ind.')

        # Cargo los archivos segun la lista
        l_objetos = list()
        for nombre_archivo in l_archivos:

            self2 = Analista(nombre=nombre_analistas, carpeta=carpeta)

            nombre_archivo = nombre_archivo.replace(".joblib", "")
            self2.cargar(carpeta=carpeta, nombre_archivo=nombre_archivo)

            l_objetos.append(self2)

        # Retorno
        return l_objetos

    def cargar_analista(self, nombre_analistas, ind):
        temp = self.cargar_versiones(nombre_analistas=nombre_analistas, ind=ind)[0]
        return temp

    def actualizar_tablero(self, nombre_analistas):

        analistas = self.cargar_versiones(nombre_analistas)

        def f(analista):
            analista.calcular_metadata()
            return analista.metadata

        metas = [f(analista) for analista in analistas]
        metas = pd.DataFrame(metas)
        self.tablero = metas

    def crear_analista(self, nombre_analistas, datamanager):

        self2 = Analista(nombre=nombre_analistas, carpeta=self.carpeta)
        self2.agregar_datamanager(datamanager)

        return self2