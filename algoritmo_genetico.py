# -*- coding: utf-8 -*-
"""
Modulo mediante el cual se configura un algortimo genetico generico aplicable
a cualquier funcion.
"""
import copy
import random
import numpy as np
import pandas as pd

from iteration_utilities import unique_everseen


class AlgoritmoGenetico:
    def __init__(
        self,
        dict_variables,
        function,
        lista_elementos=[],
        correr_algortimo=True,
        n_population=20,
        n_generations=100,
        crossover_proba=1.0,
        mutation_proba=0.5,
        crossover_independent_proba=0.1,
        mutation_independent_proba=0.05,
        tournament_size=3,
        n_random_population_each_gen=5,
        add_mutated_hall_of_fame=True,
        n_gen_no_change=5,
        valor_corte=10000000000.0,
        verbose=True,
    ):

        # Copio los parametros
        self.dict_variables = dict_variables
        self.function = function
        self.lista_elementos = lista_elementos
        self.n_population = n_population
        self.crossover_proba = crossover_proba
        self.mutation_proba = mutation_proba
        self.n_generations = n_generations
        self.crossover_independent_proba = crossover_independent_proba
        self.mutation_independent_proba = mutation_independent_proba
        self.tournament_size = tournament_size
        self.n_random_population_each_gen = n_random_population_each_gen
        self.add_mutated_hall_of_fame = add_mutated_hall_of_fame
        self.n_gen_no_change = n_gen_no_change
        self.valor_corte = valor_corte
        self.verbose = verbose

        # Creo variables que voy a utilizar
        self.historial = pd.DataFrame()
        self.historial_poblacion = []
        self.historial_fitness = []

        self.fitness_salon_de_la_fama = None
        self.poblacion_salon_de_la_fama = None
        self.mejor_individuo = None
        self.mejor_fitness = None

        # Corro el algoritmo
        if correr_algortimo:
            self.correr_algortimo()

    # =============================================================================
    # GENERAR POBLACIONES
    # =============================================================================
    def generar_parametro_aleatorio(self, k, v):
        """
        Funcion que genera un parametro aleatorio dado el nombre de la variable
        (k) y su configuracion (v).
        """
        if v["type"] is float:
            variable = np.random.uniform(v["low"], v["high"])

        elif v["type"] is int:
            variable = int(round(np.random.uniform(v["low"], v["high"]), 0))

        elif v["type"] is bool:

            if "p" in v.keys():
                variable = np.random.choice([True, False], p=v["p"])

            else:
                variable = np.random.choice([True, False])

        elif v["type"] is list:

            if "p" in v.keys():
                variable = np.random.choice(v["lista"], p=v["p"])

            else:
                variable = np.random.choice(v["lista"])

        else:
            raise TypeError('Error en la definicion de v["type"]')

        return variable

    def inicializar_poblacion(self, cantidad=None):
        """
        Funcion que genera la poblacion inicial.
        """
        # Creo la lista inicial
        population = []

        # Si la cantidad es None entonces tomo la cantidad del objeto
        if cantidad is None:
            cantidad = self.n_population

        # Genero params aleatorios para cada uno de los individuos
        for i in range(cantidad):

            variables = dict()
            for k, v in self.dict_variables.items():
                variables[k] = self.generar_parametro_aleatorio(k, v)

            population.append(variables)

        # Elimino los repetidos
        population = list(unique_everseen(population))
        return population

    # =============================================================================
    # EVALUAR
    # =============================================================================
    def calcular_funcion_objetivo(self, poblacion):
        """
        Funcion que permite evaluar la funcion objetivo sobre una poblacion
        (lista de individuos).

        Parameters
        ----------
        poblacion. list
            Lista de individuos

        Returns
        -------
            Lista de floats con el resultado de cada evaluacion.
        """
        fitness_total = [self.function(individuo) for individuo in poblacion]
        return fitness_total

    def calcular_estadisticas(self, fitness):
        """
        Funcion que permite calcular las estadisticas en base a una lista de
        resultados.

        Parameters
        ----------
        fitness. list
            Lista de floats con el resultado de cada evaluacion.

        Returns
        -------

        """
        estad = pd.DataFrame(
            [
                len(fitness),
                np.mean(fitness),
                np.std(fitness),
                np.min(fitness),
                np.max(fitness),
            ]
        ).T
        estad.columns = ["Cantidad", "Media", "Desvio", "Minimo", "Maximo"]

        hist_temp = pd.concat([self.historial, estad], ignore_index=True)

        # Calculo las generaciones en que se repite el maximo
        maximo = hist_temp.iloc[:, 4].max()
        historial = hist_temp.iloc[:, 4]
        gen_primer_maximo = historial.loc[maximo == historial].index[0]
        gen_actual = historial.index[-1]
        contador_gen = gen_actual - gen_primer_maximo

        estad = pd.DataFrame(
            [
                len(fitness),
                np.mean(fitness),
                np.std(fitness),
                np.min(fitness),
                np.max(fitness),
                int(contador_gen),
            ]
        ).T
        estad.columns = [
            "Cantidad",
            "Media",
            "Desvio",
            "Minimo",
            "Maximo",
            "n_gen_repetida",
        ]

        self.historial = pd.concat([self.historial, estad], ignore_index=True)

    # =============================================================================
    # SALON DE LA FAMA
    # =============================================================================
    def generar_index_mejores(self, cantidad, fitness):

        # Calculo el ranking
        ranking = pd.DataFrame(fitness).rank(method="min")
        ranking = ranking.sort_values(by=0, ascending=False)

        # Selecciono los mejores
        return ranking.head(cantidad).index

    def calcular_salon_de_la_fama(self, cantidad=None):
        """
        Funcion que selecciona los mejores individuos del historial de individuos
        evaluados.

        """
        # Completo la cantidad
        if cantidad is None:
            cantidad = self.tournament_size

        # rankeo el historial de fitness
        temp = (
            pd.Series(self.historial_fitness)
            .rank(ascending=False, method="first")
            .sort_values()
        )

        # Tomo los primeros del ranking
        inds = list(temp.loc[temp <= cantidad].index)

        poblacion_2 = [self.historial_poblacion[i] for i in inds]
        fitness_2 = [self.historial_fitness[i] for i in inds]

        # Actualizo el salon de la fama
        self.poblacion_salon_de_la_fama = poblacion_2
        self.fitness_salon_de_la_fama = fitness_2

    # =============================================================================
    # CRUCE Y MUTACION
    # =============================================================================
    def cruzar(self, poblacion):
        """
        Funcion que cruza los genes de una poblacion y da como resultado una nueva
        poblacion modificada respecto a la que se pasó como parametro.

        El metodo consiste en tomar los individuos de la poblacion inicial e intercambiar
        elementos con otros individuos de la poblacion aleatoriamente.

        Parameters
        ----------
        poblacion. list
            Lista de individuos

        Returns
        -------
        list.
            Lista de individuos cruzados

        """
        # Copio el objeto
        poblacion_copy = copy.deepcopy(poblacion)

        # Calculo las probas de cruce
        f_atributo_c = self.obtener_valor_atributo_random("crossover_proba")
        f_atributo_ci = self.obtener_valor_atributo_random(
            "crossover_independent_proba"
        )

        # selecciono los que van a participar del crossover
        sel_crossover = [
            individuo
            for individuo in poblacion_copy
            if random.uniform(0, 1) < f_atributo_c
        ]

        if len(sel_crossover) < 2:
            sel_crossover = [
                random.choice(poblacion_copy),
                random.choice(poblacion_copy),
            ]

        # Selecciono para cada individuo
        nuevos = []
        for nuevo in sel_crossover:  # nuevo = poblacion_copy[0]

            nuevo_2 = {}

            # Tomo aleatoriamente uno de los candidatos
            padre = random.choice(sel_crossover)

            # Selecciono para cada etiqueta
            for etiqueta in self.dict_variables.keys():
                if random.uniform(0, 1) < f_atributo_ci:
                    nuevo_2[etiqueta] = padre[etiqueta]
                else:
                    nuevo_2[etiqueta] = nuevo[etiqueta]

            nuevos.append(nuevo_2)

        nuevos = list(unique_everseen(nuevos))

        return nuevos

    def mutar(self, poblacion):
        """
        Funcion que modifica (muta) los genes de una poblacion y da como resultado una nueva
        poblacion modificada respecto a la que se paso como parametro.

        El metodo consiste en tomar los individuos de la poblacion inicial y cambiar aleatoreamente
        los elementos de cada uno.

        Parameters
        ----------
        poblacion. list
            Lista de individuos

        Returns
        -------
        list.
            Lista de individuos mutados.

        """
        # Copio el objeto
        poblacion_copy = copy.deepcopy(poblacion)

        # calculo las probabilidades de mutacion
        f_atributo_m = self.obtener_valor_atributo_random("crossover_proba")
        f_atributo_mi = self.obtener_valor_atributo_random(
            "crossover_independent_proba"
        )

        # Selecciono para cada individuo
        nuevos = []
        for nuevo in poblacion_copy:  # nuevo = poblacion_copy[0]

            nuevo_2 = {}
            if random.uniform(0, 1) < f_atributo_m:

                # Selecciono para cada etiqueta
                for k, v in self.dict_variables.items():

                    # generando ese valor y lo inserto en el nuevo individuo
                    if random.uniform(0, 1) < f_atributo_mi:
                        nuevo_2[k] = self.generar_parametro_aleatorio(k, v)
                    else:
                        nuevo_2[k] = nuevo[k]

            nuevos.append(nuevo_2)

        nuevos = list(unique_everseen(nuevos))

        return nuevos

    # =============================================================================
    # Corro el algoritmo
    # =============================================================================
    def correr_algortimo(self):
        """
        Metodo con el cual se corre el proceso de algoritmo genetico.

        El metodo consiste en dos partes. Una primera parte en donde se evaluan
        una poblacion aleatoria o que haya sido pasada como parametro en la
        iniciacion del elemento.

        Una vez corrido este se evaluan y se toman los k-mejores, sobre los cuales
        se aplican los metodos mutar y cruzar, obteniendose una nueva poblacion a
        evaluar. Además se pueden sumar nuevos individuos generados aleatoriamente.

        Este proceso se repite iterativamente hasta que se produce el corte del proceso,
        el cual puede suceder por diferentes motivos: alcanza el maximo de iteraciones,
        no se obtiene un mejor fitness luego de una determinada cantidad de interaciones
        o se alcanza un valor determinado.

        """
        # Genero la poblacion inicial
        poblacion_inicial = self.inicializar_poblacion()
        poblacion_inicial.extend(self.lista_elementos)

        # Calculo el fitness
        fitness_inicial = self.calcular_funcion_objetivo(poblacion=poblacion_inicial)

        # Storeo
        self.historial_poblacion.extend(poblacion_inicial)
        self.historial_fitness.extend(fitness_inicial)

        # Calculo las estadisticas y las imprimo
        self.calcular_estadisticas(fitness=fitness_inicial)

        # Imprimo en pantalla
        if self.verbose:
            print("=" * 60)
            print("Resultado de la Generacion 0")
            print("=" * 60)
            print(self.historial)
            print("\n" * 5)

        # Calculo el salon de la fama
        self.calcular_salon_de_la_fama()

        for gen in range(self.n_generations):  # gen = 0

            # Corto si no hay elementos en el salon de la fama
            if len(self.poblacion_salon_de_la_fama) <= 1:
                print(
                    "Finalizado antes de tiempo por tener solo un individuo para cruzar."
                )
                break

            # Creo la lista objetivo
            nuevos = []

            # Armo el cruce y mutacion de los mejores
            nuevos_cruzados = self.cruzar(poblacion=self.poblacion_salon_de_la_fama)
            nuevos.extend(nuevos_cruzados)

            nuevos_mutados = self.mutar(poblacion=nuevos_cruzados)
            nuevos.extend(nuevos_mutados)

            # Genero individuos aleatorios nuevos
            if self.n_random_population_each_gen:
                cantidad = self.n_random_population_each_gen
                nuevos_random = self.inicializar_poblacion(cantidad=cantidad)
                nuevos.extend(nuevos_random)

            # Muto el salon de la fama
            if self.add_mutated_hall_of_fame:
                nuevos_sldf = self.mutar(poblacion=self.poblacion_salon_de_la_fama)
                nuevos.extend(nuevos_sldf)

            nuevos_unicos = list(unique_everseen(nuevos))
            poblacion_nuevos = [
                nuevo
                for nuevo in nuevos_unicos
                if nuevo not in self.historial_poblacion
            ]

            # Calculo de nuevo sobre la siguiente generacion
            fitness_nuevos = self.calcular_funcion_objetivo(poblacion=poblacion_nuevos)
            assert len(fitness_nuevos) == len(poblacion_nuevos)

            # Storeo
            self.historial_poblacion.extend(poblacion_nuevos)
            self.historial_fitness.extend(fitness_nuevos)

            # Calculo las estadisticas
            self.calcular_estadisticas(fitness=fitness_nuevos)

            print("=" * 60)
            print(f"Resultado de la Generacion {gen+1}")
            print("=" * 60)
            print(self.historial)
            print("\n" * 5)

            # armo el salon de la fama
            self.calcular_salon_de_la_fama()

            # Calculo las generaciones en que se repite el maximo
            maximo = self.historial.iloc[:, 4].max()
            historial = self.historial.iloc[:, 4]
            gen_primer_maximo = historial.loc[maximo == historial].index[0]
            gen_actual = historial.index[-1]
            contador_gen = gen_actual - gen_primer_maximo

            # Corto el proceso si se cumple alguna condicion
            if contador_gen >= self.n_gen_no_change:
                break
            if maximo >= self.valor_corte:
                break

        # Tomo el mejor de todos los individuos
        ind = list(
            self.generar_index_mejores(
                cantidad=1, fitness=self.fitness_salon_de_la_fama
            )
        )
        self.mejor_individuo = self.poblacion_salon_de_la_fama[ind[0]]
        self.mejor_fitness = self.fitness_salon_de_la_fama[ind[0]]

    # =========================================================================
    # OTRAS FUNCIONES
    # =========================================================================
    def obtener_valor_atributo_random(self, nombre_atributo):
        """
        Metodo qu epermite calcular el valor de un atributo de forma aletatoria,
        el cual estara compredido en una lista.

        Parameters
        ----------
        nombre_atributo. str
            Nombre del atributo que va a calcularse aleatoriamente.

        Returns
        -------

        """
        # Tomo una etiqueta aleatoria de la seleccion y lo inserto
        atributo = getattr(self, nombre_atributo)

        if type(atributo) == list:
            f_atributo = random.uniform(atributo[0], atributo[1])

        elif type(atributo) == float:
            f_atributo = atributo

        else:
            raise TypeError(
                f"Error en la definicion del {nombre_atributo}. Debe ser float o list de dos elementos."
            )

        return f_atributo
