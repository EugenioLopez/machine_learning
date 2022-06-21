# -*- coding: utf-8 -*-
"""
Modulo en el que se genera un parser que servira para la carga de datos.
"""
import os
import pickle
import copy
import requests
import numpy as np
import pandas as pd
import joblib


class Parser:
    """
    Clase con la cual se parametriza un parser que servirá para descargar,
    guardar en disco y cargar del disco distintos tipos de archivos y llevar
    un control sobre estos.

        Parametros:

        De entrada:

            nombre. string. Obligatorio.
                Nombre del parser.

            carpeta. string. Obligatorio.
                Carpeta principal del parser donde se guardaran y cargaran
                todos los archivos y variables por defecto.

    """

    def __init__(self, nombre, carpeta):

        # Copio las carpetas
        self.nombre = nombre
        self.carpeta = carpeta

        # Completo en caso de que no tenga las barras al final
        if carpeta[-2:] != '//':
            self.carpeta += '//'

        # Armo los dataframes para las estadisticas
        self.archivos_descargados = pd.DataFrame()
        self.variables_descargadas = pd.DataFrame()

    # =========================================================================
    # METODOS DE GESTION DEL PARSER
    # =========================================================================
    # Funciones para guardar y cargar archivos pickle

    def pickleo(
        self,
        nombre_variable,
        nombre_archivo,
        comprimir=False,
        carpeta=None,
        usar_joblib=True,
        grupo="variables descargadas",
    ):

        # Cargo el nombre de la carpeta de la descarga
        if not carpeta:
            carpeta = self.carpeta

        # Completo si no tiene las barras al final
        if carpeta[-2:] != '//':
            carpeta += '//'

        # armo el directorio si no existe
        if not os.path.isdir(carpeta):
            os.mkdir(carpeta)

        # Para el caso joblib comprimido
        if usar_joblib and comprimir:
            nombre_archivo = f"{carpeta}{nombre_archivo}.gz"
            joblib.dump( nombre_variable, nombre_archivo, compress=comprimir)

        # Para el caso joblib sin comprimir
        elif usar_joblib and not comprimir:
            nombre_archivo = f"{carpeta}{nombre_archivo}.joblib"
            joblib.dump( nombre_variable, nombre_archivo)

        # Para los demás casos uso pickle
        else:
            nombre_archivo = f"{carpeta}{nombre_archivo}.p",
            pickle.dump(nombre_variable, open(nombre_archivo, "wb"))

        # Registro en el log
        self.variables_descargadas = self.agregar_log_url(
            df_log=self.variables_descargadas,
            url="",
            grupo=grupo,
            nombre_archivo=nombre_archivo,
            nombre_variable="",
            fecha_hoy=True,
        )

    def unpickleo(self, nombre_archivo, usar_joblib=True, carpeta=None):
        """
        Función que permite cargar el objeto en la carpeta seleccionada en el
        disco.

        Parametros
        ----------

            nombre_archivo. string. Obligatorio
                Nombre con que se guardara el archivo.

            usar_joblib. bool. Optativo. Default: True
                Parametro que determina si se usa joblib o no.

            carpeta. {None, string}. Optativo. Default: None
                Nombre de la carpeta donde se guardara el archivo. En caso de
                ser None guarda en la carpeta principal del parser.

        De salida: variable.
                Variable cargada.

        """
        if carpeta is None:
            carpeta = self.carpeta

        if carpeta[-2:] != '//':
            carpeta += '//'

        if usar_joblib:
            nombre_archivo = f"{carpeta}{nombre_archivo}.joblib"
            return joblib.load(nombre_archivo)

        else:
            nombre_archivo = f"{carpeta}{nombre_archivo}.p"
            return pickle.load(nombre_archivo, "rb")

    def unpicklear_todo(self, patron="joblib"):
        nombre_archivos = [s for s in os.listdir(self.carpeta) if patron in s]
        return [self.unpickleo(n[:-7]) for n in nombre_archivos]

    def copiar(self, otro):
        """
        Funcion que permite copiar las caracteristicas un otro parser en el
        objeto parser que lo ejecuta.

        Parametros:

        De Entrada:

            otro. Parser. Obligatorio.
                instancia de la clase parser que desea copiarse.

        De salida: -
        """
        self.nombre = copy.deepcopy(otro.nombre)
        self.carpeta = copy.deepcopy(otro.carpeta)
        self.archivos_descargados = copy.deepcopy(otro.archivos_descargados)
        self.variables_descargadas = copy.deepcopy(otro.variables_descargadas)

    def exportar(self, carpeta=None, nombre=None):
        """
        Funcion que permite exportar los datos del parser a un archivo excel.
        El archivo se localiza en la carpeta principal del parser.

        Parametros

        De entrada:

            carpeta. string. Optativo. Default: None
                string con la carpeta donde se alojara el archivo exportado.
                en caso de ser None lo exporta a la carpeta principal del
                parser.

            nombre. string. Optativo. Default: None
                string con el nombre del archivo exportado.
                en caso de ser None lo exporta con el nombre del parser.

        De salida: -

        """
        # ARMO LOS PARAMETROS POR DEFECTO
        if carpeta is None:
            carpeta = self.carpeta

        if nombre is None:
            nombre = self.nombre

        # ARMO LA RUTA DEL ARCHIVO
        ruta_guardar = f"{carpeta}{nombre}.xlsx"

        # INICIALIZO EL WRITER
        excel_writer = pd.ExcelWriter(
            ruta_guardar, datetime_format="dd mmm YYYY, HH:mm:ss"
        )

        # EXPORTOR LOS ARCHIVOS DESCARGADOS
        self.archivos_descargados.to_excel(
            excel_writer, sheet_name="archivos_descargados"
        )

        # EXPORTO LAS VARIABLES ARMADAS
        self.variables_descargadas.to_excel(
            excel_writer, sheet_name="variables_descargadas"
        )

        excel_writer.save()

    def agregar_log_url(
        self,
        df_log,
        url=None,
        grupo=None,
        nombre_archivo=None,
        nombre_variable=None,
        fecha_hoy=False,
    ):
        """
        Funcion que agrega el objeto que se esta trabajando al dataframe log
        del parser.

        Parametros:

        De entrada:

            df_log: pandas.DataFrame. Obligatorio.
                Dataframe que servira de registro de log.

            url: string. Optativo. default: None
                Url que va a registrarse en el log.

            grupo: string. Optativo. default: None
                Grupo al que pertenece el registro, que va a registrarse
            en el log.

            nombre_archivo: string. Optativo. default: None
                Nombre del archivo que va a registrarse en el log.

            nombre_variable: string. Optativo. default: None
                Nombre de la variable que va a registrarse en el log.

            fecha_hoy: bool. Optativo. default: True
                variable booleana que en caso de ser True registra la fecha
                de hoy. Si es falso registra un valo NaN.

        De Salida: pandas.DataFrame.
                df_log con los datos actualizados.

        """

        # Genero la fecha en bsae a los parametros pasados
        if fecha_hoy:
            fecha = pd.Timestamp.today()
        else:
            fecha = np.nan

        # Genero el nombre del archivo y de la variable
        if nombre_archivo is None:
            nombre_archivo = url[url.rfind("/") + 1 :]

        if nombre_variable is None:
            nombre_variable = url[url.rfind("/") + 1 : url.rfind(".")]

        # Armo el DataFrame lo inserto en el control de descargas
        archivo = pd.DataFrame(
            {
                "grupo": [grupo],
                "url": [url],
                "nombre_archivo": [nombre_archivo],
                "nombre_variable": [nombre_variable],
                "fecha": fecha,
                "ruta": [f"{self.carpeta}{nombre_archivo}"],
            }
        )

        df_log = pd.concat([df_log, archivo])

        # Elimino los duplicados manteniendo el ultimo cargado
        df_log.drop_duplicates(keep="last", subset=["nombre_archivo"], inplace=True)

        return df_log

    # ==========================================================================
    # METODOS PARA LA DESCARGA DE ARCHIVOS
    # ==========================================================================
    def agregar_urls(
        self,
        lista_urls,
        grupo=None,
        nombre_archivo=None,
        fecha_hoy=False,
        descargar=False,
    ):
        """
        Función que permite agregar una lista de urls al log del parser.

        Parametros:

        De entrada:

            lista_urls. lista. Obligatorio.
                Lista de urls que van a agregarse al log.

            grupo. string. Opcional. Default: None
                Nombre del grupo al que pertenecen las urls agregadas.

            fecha_hoy. bool. Opcional. Default: False
                parametro que determina si se registra en el log la fecha
                de hoy.

            descargar. bool. Opcional. Default: False
                Parametro que determina si la url se descarga al disco o no.
        """

        # Descargo en caso de corresponder
        if descargar:

            for url in lista_urls:

                if nombre_archivo is None:
                    nombre_archivo = url[url.rfind("/") + 1 :]

                # Descargo
                self.descargar_url(url, nombre_archivo=url[url.rfind("/") + 1 :])
                self.archivos_descargados = self.agregar_log_url(
                    self.archivos_descargados,
                    url=url,
                    grupo=grupo,
                    nombre_archivo=nombre_archivo,
                    fecha_hoy=True,
                )

        else:

            for url in lista_urls:

                self.archivos_descargados = self.agregar_log_url(
                    self.archivos_descargados,
                    url=url,
                    grupo=grupo,
                    nombre_archivo=url[url.rfind("/") + 1 :],
                    fecha_hoy=fecha_hoy,
                )

    def descargar_url(self, url, nombre_archivo=None):
        """
        Funcion con la que descargo el archivo y lo alojo en memoria. El
        archivo se guardara en la carpeta del parser.

        Parametros:

        De entrada:

            url: string. Obligatorio.
                Url del archivo a descargar

            nombre_archivo: string. Optcional. Default: None
                Nombre del archivo que va a descargarse. En caso de ser None
                sera el nombre del archivo que identifique en la url

        """
        if nombre_archivo is None:
            nombre_archivo = url[url.rfind("/") + 1 :]

        # Descargo
        print(f"descargando {nombre_archivo}")
        myfile = requests.get(url)
        open(f"{self.carpeta}{nombre_archivo}", "wb").write(myfile.content)

    # ==========================================================================
    # METODOS FRONT PARA GESTION DE ARCHIVOS
    # ==========================================================================
    def descargar_archivos(self, lista_urls, grupo):
        """
        Funcion que permite descargar archivos.

        Parametros:

        De entrada:

            lista_urls. lista. Obligatorio
                Lista de urls que van a agregarse al log.

            grupo. string. Opcional. Default: None
                Grupo al que pertenecen los archivos que se estan descargando.
        """
        self.agregar_urls(lista_urls, grupo=grupo, descargar=True)

    def agregar_archivos(self, lista_urls, grupo=None, fecha_hoy=False):
        """
        Funcion que permite agregar archivos al logo del parser.

        Parametros:

        De entrada:

            lista_urls. lista. Obligatorio
                Lista de urls que van a agregarse al log.

            grupo. string. Opcional. Default: None
                Grupo al que pertenecen los archivos que se estan agregando.

            fecha_hoy. Opcional. Default: False
                Parametros que en caso de ser True agrega la fecha en el log.
                Caso contrario agrega un valor nan.

        De salida: -
        """
        # Agrego las urls sin descargar
        self.agregar_urls(lista_urls, grupo=grupo, fecha_hoy=fecha_hoy, descargar=False)

    def actualizar_archivos(self, lista_grupos=None):
        """
        Función que permite bajar nuevamente los archivos del log.

        Parametros:

        De entrada:

            lista_grupos: {None, lista}. Optativo. Default: None
                Parametro que sirve para determinar que grupos del parser se
                actualizan. Si es None se actualizan todos los grupos.

        """
        # Cargo los grupos dentro del parser
        if lista_grupos is None:
            lista_grupos = pd.unique(list(self.archivos_descargados["grupo"]))

        for grupo in lista_grupos:

            lista_urls = self.archivos_descargados.loc[
                self.archivos_descargados["grupo"] == grupo
            ]["url"]

            # Agrego las urls pero con el parametro descarga habilitado
            self.descargar_archivos(lista_urls, grupo=grupo)

    # ==========================================================================
    # METODOS PARA GESTIONAR VARIABLES
    # ==========================================================================
    def generar_diccionario_series(
        self, funcion, parametros, ruta=None, grupo=None, pickleo=True
    ):
        """
        Función que permite generar los diccionarios de series y guardarlos
        en disco.

        Parametros

        De entrada:

            funcion. function. Obligatorio
                Función que devuelve un diccionario de series que va a ser
                guardado.

            parametros. dict. Obligatorio
                Parametros para la funcion que va a generar el diccionario.

            ruta: string. Optativo. default: None
                Ruta que va a registrarse en el log.

            grupo: string. Optativo. default: None
                Grupo que va a registrarse en el log.

            pickleo. bool. Default: True
                Parametro que en caso de ser verdadero permitira guardar el
                diccionario en disco, en la carpeta del parser.

        De salida. dict.
                Diccionario con las series generadas.
        """

        # Hago el split del archivo
        diccionario_series = funcion(**parametros)

        if pickleo:
            self.picklear_diccionario(diccionario_series, ruta=ruta, grupo=grupo)

        return diccionario_series

    def picklear_diccionario(self, diccionario_series, ruta=np.nan, grupo=np.nan):
        """
        Función que permite guardar un diccionario de variables en disco.

        Parametros

        De entrada:

            diccionario_series. dict. Obligatorio.
                Diccionario que va a guardarse en el disco.

            ruta. string. Optativo. Default: np.nan.
                Ruta del diccionario

            grupo. string. Optativo. Default: np.nan.
                grupo al que pertenece la variable que se picklea.

        """
        # Resgistro en el log y guardo
        for nombre, serie in diccionario_series.items():

            # Guardo
            try:
                self.pickleo(serie, nombre, grupo=grupo)
            except OSError:
                pass

    def generar_diccionario_desde_log(
        self, funcion, grupo, parametros_adicionales=dict(), pickleo=True
    ):
        """
        Funcion que permite aplicar una funcion a un mismo grupo de archivos
        descargados.

        Parámetros:

        De entrada:

            funcion. function. Obligatorio
                Funcion que se va a aplicar al grupo. Esta tiene que
                tener determinadas caracteristicas.

                parametros de entrada: debe llevar obligatoriamente el
                                       parametro ruta.
                                       debe llevar obligatorialemente el
                                       parametro **kwargs.

                parametros de salida: debe devolver un diccionario de series.

            grupo. string. Obligatorio.
                string con el nombre del grupo al que se le aplicara la
                funcion.

            parametros adicionales: dict. Optativo: default: dict()
                Son los parametros adicionales a la funcion (exceptuando ruta).
                Si la funcion posee parametros adicionales estos deben estar
                incluidos en este diccionario.

            pickleo. bool. Optional. Default: True
                Parametro que determina si el diccionario se guarda o no.
                En caso de guardarse se guarda en la carpeta principal del
                parser.
        """

        filt = self.archivos_descargados["grupo"] == grupo
        lista_dict_parametros = self.archivos_descargados.loc[filt].to_dict(
            orient="records"
        )

        dict_series = {}

        for parametros_fila in lista_dict_parametros:

            # Junto los parametros
            parametros_fila.update(parametros_adicionales)

            # Aplico la funcion generar_diccionario_series
            dict_series.update(
                self.generar_diccionario_series(
                    funcion=funcion,
                    parametros=parametros_fila,
                    ruta=parametros_fila["ruta"],
                    grupo=grupo,
                    pickleo=pickleo,
                )
            )

        return dict_series

    def exportar_df(self, df, nombre_archivo, carpeta=None):

        # Completo la ruta
        if carpeta is None:
            carpeta = self.carpeta

        # Creo la ruta al archivo
        ruta_guardar = carpeta + nombre_archivo + ".xlsx"

        # Creo el writer
        excel_writer = pd.ExcelWriter(ruta_guardar)

        # Cargo el DF
        if type(df) == pd.core.frame.DataFrame or type(df) == pd.core.frame.Series:
            df.to_excel(excel_writer)

        elif type(df) == list:
            for i, v in enumerate(df):
                v.to_excel(excel_writer, sheet_name=str(i))

        elif type(df) == dict:
            for k, v in df.items():
                v.to_excel(excel_writer, sheet_name=str(k))
        else:
            raise SyntaxError(
                "Error. El parametro df puede ser un pandas.Dataframe, lista o dict."
            )

        # Guardo
        excel_writer.save()

    def importar_df_excel(self, carpeta, nombre_archivo):

        # Completo la ruta
        if carpeta is None:
            carpeta = self.carpeta

        # Creo la ruta al archivo
        ruta_guardar = carpeta + nombre_archivo + ".xlsx"

        # Importo
        df = pd.read_excel(ruta_guardar)

        return df

    def exportar_csv(self, df, nombre_archivo, carpeta=None):

        # Completo la ruta
        if carpeta is None:
            carpeta = self.carpeta

        # Creo la ruta al archivo
        ruta_guardar = carpeta + nombre_archivo + ".csv"

        # Exporto
        # Cargo el DF
        if type(df) == pd.core.frame.DataFrame or type(df) == pd.core.frame.Series:
            df.to_csv(ruta_guardar, index=True)

        elif type(df) == list:
            for i, v in enumerate(df):
                v.df.to_csv(ruta_guardar, index=True)

        elif type(df) == dict:
            for k, v in df.items():
                v.df.to_csv(ruta_guardar, index=True)
        else:
            raise SyntaxError(
                "Error. El parametro df puede ser un pandas.Dataframe, lista o dict."
            )

