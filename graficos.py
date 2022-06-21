# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:12:26 2019

@author: Olivia
"""
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px

import plotly.io as pio
pio.renderers.default = "browser"

# %%===========================================================================
#
# =============================================================================
def multiple_plots(grafs, cantH=2, cantV=1, ancho=15, alto=6, hspace=None):

    fig, ax = plt.subplots(cantH, cantV, figsize=(ancho, alto))
    fig.subplots_adjust(hspace=hspace)
    ax[0].add_subplot(grafs[0])

    plt.show(fig)

    return fig


# %%===========================================================================
# PLOT
# =============================================================================
def plot(X,
         style='seaborn',
         label_y=None,
         label_title=None,
         fontsize_title=15,
         stacked=True,
         save=False,
         save_ruta=None,
         save_format='pdf',
         **params):


    plt.style.use(style)
    X.plot(**params)

    plt.xlabel("Fecha")
    plt.ylabel(label_title)
    plt.title(label_title, fontsize=fontsize_title)
    plt.legend(X.columns, loc='lower left')

    if save:
        nombre = ''.join([save_ruta, label_title, f'.{label_title}'])
        plt.savefig(nombre, bbox_inches='tight')

    plt.show()
    plt.close()

    return plt


# %%===========================================================================
# HEAT MAP
# =============================================================================
def heatmap(df,
            titulo,
            ancho=800,
            alto=800,
            zmin=-1.,
            zmid=0,
            zmax=1.,
            colorscale="Inferno",
            graficar=True):

    fig = go.Figure(data=go.Heatmap(z=df.values,
                                    x=df.index,
                                    y=df.columns,
                                    zmin=zmin,
                                    zmid=zmid,
                                    zmax=zmin,
                                    colorscale=colorscale,
                                    hoverongaps=False))

    fig.update_layout(showlegend=False,
                      width=ancho,
                      height=alto,
                      autosize=False,
                      title=titulo)

    if graficar:
        fig.show()

    return fig


# =============================================================================
# GRAFICOS PLOTLY
# =============================================================================
def grafico_con_slide(list_dict_plots,
                      titulo="Serie",
                      tipo='Scatter',
                      graficar=True):

    # Create figure
    fig = go.Figure()

    for dicti in list_dict_plots:

        if tipo == 'Scatter':
            fig.add_trace(go.Scatter(**dicti))
        elif tipo == 'Box':
            fig.add_trace(go.Box(**dicti))
        else:
            raise('Tipo de grafico no configurado')

    # Set title
    fig.update_layout(
        title_text=titulo
    )

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                                dict(count=1,
                                     label="1m",
                                     step="month",
                                     stepmode="backward"),
                                dict(count=6,
                                     label="6m",
                                     step="month",
                                     stepmode="backward"),
                                dict(count=1,
                                     label="YTD",
                                     step="year",
                                     stepmode="todate"),
                                dict(count=1,
                                     label="1y",
                                     step="year",
                                     stepmode="backward"),
                                dict(step="all")
                                ])
                                ),
                        rangeslider=dict(
                            visible=True
                        ),
                        type="date"
                    )
                    )

    if graficar:
        fig.show()

    return fig

# =============================================================================
# GRAFICO DE CANDLESTICK
# =============================================================================
def plot_candlestick(df,
                     add_ma=False,
                     add_bbands=False,
                     INCREASING_COLOR='#17BECF',
                     DECREASING_COLOR='#7F7F7F'):

    # Initial candlestick chart
    data = [dict(type='candlestick',
                 open=df.Open,
                 high=df.High,
                 low=df.Low,
                 close=df.Close,
                 x=df.index,
                 yaxis='y2',
                 name='GS',
                 increasing=dict(line=dict(color=INCREASING_COLOR)),
                 decreasing=dict(line=dict(color=DECREASING_COLOR)),
                 )
            ]

    layout = dict()

    fig = dict(data=data, layout=layout)

    fig['layout'] = dict()
    fig['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig['layout']['xaxis'] = dict(rangeselector=dict(visible=True))
    fig['layout']['yaxis'] = dict(domain=[0, 0.2], showticklabels=False)
    fig['layout']['yaxis2'] = dict(domain=[0.2, 0.8])
    fig['layout']['legend'] = dict(orientation='h', y=0.9, x=0.3, yanchor='bottom')
    fig['layout']['margin'] = dict(t=40, b=40, r=40, l=40)

    rangeselector=dict(
                        visible=True,
                        x=0, y=0.9,
                        bgcolor='rgba(150, 200, 250, 0.4)',
                        font=dict(size=13),
                        buttons=list([
                            dict(count=1,
                                 label='reset',
                                 step='all'),
                            dict(count=1,
                                 label='1yr',
                                 step='year',
                                 stepmode='backward'),
                            dict(count=3,
                                 label='3 mo',
                                 step='month',
                                 stepmode='backward'),
                            dict(count=1,
                                 label='1 mo',
                                 step='month',
                                 stepmode='backward'),
                            dict(step='all')
                        ]))

    fig['layout']['xaxis']['rangeselector'] = rangeselector

    if add_ma:
        def movingaverage(interval, window_size=10):
            window = np.ones(int(window_size))/float(window_size)
            return np.convolve(interval, window, 'same')


        mv_y = movingaverage(df.Close)
        mv_x = list(df.index)

        # Clip the ends
        mv_x = mv_x[5:-5]
        mv_y = mv_y[5:-5]

        fig['data'].append( dict( x=mv_x, y=mv_y, type='scatter', mode='lines', 
                                line = dict( width = 1 ),
                                marker = dict( color = '#E377C2' ),
                                yaxis = 'y2', name='Moving Average' ) )

    colors = []

    for i in range(len(df.Close)):
        if i != 0:
            if df.Close[i] > df.Close[i-1]:
                colors.append(INCREASING_COLOR)
            else:
                colors.append(DECREASING_COLOR)
        else:
            colors.append(DECREASING_COLOR)

    fig['data'].append( dict( x=df.index, y=df.Volume,                         
                            marker=dict( color=colors ),
                            type='bar', yaxis='y', name='Volume' ) )

    # Agrego bandas de bollinger
    if add_bbands:
        def bbands(price, window_size=10, num_of_std=5):
            rolling_mean = price.rolling(window=window_size).mean()
            rolling_std  = price.rolling(window=window_size).std()
            upper_band = rolling_mean + (rolling_std*num_of_std)
            lower_band = rolling_mean - (rolling_std*num_of_std)
            return rolling_mean, upper_band, lower_band

        bb_avg, bb_upper, bb_lower = bbands(df.Close)

        fig['data'].append( dict( x=df.index, y=bb_upper, type='scatter', yaxis='y2', 
                                line = dict( width = 1 ),
                                marker=dict(color='#ccc'), hoverinfo='none', 
                                legendgroup='Bollinger Bands', name='Bollinger Bands') )

        fig['data'].append( dict( x=df.index, y=bb_lower, type='scatter', yaxis='y2',
                                line = dict( width = 1 ),
                                marker=dict(color='#ccc'), hoverinfo='none',
                                legendgroup='Bollinger Bands', showlegend=False ) )

    fig2 = go.Figure(fig['data'])
    fig2 = fig2.update_layout(fig['layout'])
    fig2.show()

    return fig2


def distplot_superpuetos(paper_data,
                         highest_mean_return,
                         titulo,
                         bin_size=.005):

    a = []
    b = []
    for p in highest_mean_return:
        a.append(paper_data.loc[:, p].dropna())
        b.append(p)

    fig = ff.create_distplot(a,
                             b,
                             bin_size=.005,
                             curve_type='normal')

    fig['layout'].update(title=titulo)
    fig.show()


# ============================================================================
# GRAFICO LA TABLA
# =============================================================================
def graficar_tabla(report_df_pre,
                   titulo=None,
                   graficar=True,
                   formato_decimal='{0:.4f}',
                   formato_indice='%d-%m-%Y',
                   ancho=1500,
                   alto=400,
                   color_encabezado='rgb(115, 115, 115)',
                   tamano_letra_encabezado=14,
                   alineacion_encabezado='center',
                   color_indice='rgb(189, 189, 189)',
                   color_valores='lavender',
                   margen_superior=20,
                   margen_inferior=20,
                   margen_izquierdo=20,
                   margen_derecho=20):

     # https://plotly.com/python/builtin-colorscales/

    # HAGO UNA DEEP COPY DEL DATAFRAME
    report_df = copy.deepcopy(report_df_pre)

    # HAGO UNA DEEP COPY DEL DATAFRAME
    if type(report_df) is pd.core.series.Series:
        report_df = pd.DataFrame(report_df)

    # ASIGNO EL FORMATO DECIMAL
    if formato_decimal is not None:
        report_df = report_df.applymap(lambda x: formato_decimal.format(x))

    # CAMBIO EL FORMATO DEL INDEX Y LO SACO DEL INDEX
    if type(report_df.index) is pd.core.indexes.datetimes.DatetimeIndex:
        report_df.index = report_df.index.strftime(formato_indice)

    report_df.reset_index(drop=False, inplace=True)

    # ARMO EL GRAFICO DE TABLA
    fig = go.Figure(data=[go.Table(
                                    header=dict(values=list(report_df),
                                                font=dict(color=['rgb(45, 45, 45)'],
                                                            size=tamano_letra_encabezado),
                                                fill=dict(color=color_encabezado),
                                                align=alineacion_encabezado),

                                    cells=dict(values=report_df.values.T,
                                               fill=dict(color=[color_indice, color_valores]),
                                               align=['right', 'center'])
                                    )])

    # ARMO EL LAYOUT
    if titulo is not None:
        margen_superior += 40

    fig.update_layout(width=ancho,
                      height=alto,
                      title_text=titulo,
                      margin=dict(autoexpand=True,
                                  b=margen_inferior,
                                  l=margen_izquierdo,
                                  r=margen_derecho,
                                  t=margen_superior))

    if graficar:
        fig.show()

    return fig
