import dash_ag_grid as dag
import dash_bootstrap_components as dbc  # pip install dash-bootstrap-components
import dash_daq as daq
import matplotlib  # pip install matplotlib
import numpy as np
import pandas as pd  # pip install pandas
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html  # pip install dash
from plotly.graph_objects import Layout

from data_preparation import get_model_and_visits, get_reservations_from_model

matplotlib.use("agg")
import base64
from io import BytesIO

import matplotlib.pyplot as plt

logreg, df_visits = get_model_and_visits()
cum_visits = np.flip(np.cumsum(np.flip(df_visits.values)))

TODAY_LEAD = 75


def sigmoid(CPC):
    return 10 * (1 - np.exp(-5 * CPC))


# df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/solar.csv")
# Dropdown(
#     id="category",
#     value="Number of Solar Plants",
#     clearable=False,
#     options=df.columns[1:],
# )
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container(
    [
        html.H1(
            "Demo Campaigns",
            className="mb-2",
            style={"textAlign": "center"},
        ),
        # first row: slider ADR
        dbc.Row(
            [
                # slider of the ADR
                dbc.Col(
                    [
                        html.H3(
                            "Average Daily Rate",
                            className="mb-2",
                            style={"textAlign": "center"},
                        ),
                        # slider of the ADR
                        dcc.Slider(
                            id="ADR_slider",
                            min=50,
                            max=350,
                            step=10,
                            value=180,
                            marks={i: str(i) for i in range(50, 351, 50)},
                            tooltip={"placement": "bottom", "always_visible": True}
                            # handleLabel={"showCurrentValue": True, "label": "VALUE"},
                        ),
                    ],
                    width={"size": 4, "offset": 0},
                ),
                # slider of the CPC
                dbc.Col(
                    [
                        html.H3(
                            "Cost per Click",
                            className="mb-2",
                            style={"textAlign": "center"},
                        ),
                        # slider of the ADR
                        dcc.Slider(
                            id="CPC_slider",
                            min=0,
                            max=1,
                            step=0.1,
                            value=0.1,
                            marks={
                                f"{i:.1f}": f"{i:.1f}" for i in np.arange(0, 1.1, 0.1)
                            },
                            tooltip={"placement": "bottom", "always_visible": True}
                            # handleLabel={"showCurrentValue": True, "label": "VALUE"},
                        ),
                    ],
                    width={"size": 4, "offset": 0},
                ),
                dbc.Col(
                    [
                        html.H3(
                            "Campaign start",
                            className="mb-2",
                            style={"textAlign": "center"},
                        ),
                        # slider of the ADR
                        dcc.Slider(
                            id="Ad_start",
                            min=0,
                            max=75,
                            step=1,
                            value=60,
                            tooltip={"placement": "bottom", "always_visible": True},
                            marks={i: str(i) for i in range(0, 76, 10)},
                        ),
                    ],
                    width={"size": 4, "offset": 0},
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [dcc.Graph(id="plot1-visits", figure={})],
                                    width=6,
                                    align="center",
                                ),
                                dbc.Col(
                                    [dcc.Graph(id="plot2-CPC", figure={})],
                                    width=6,
                                    align="center",
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [dcc.Graph(id="plot3-margin", figure={})],
                                    width=6,
                                    align="center",
                                ),
                                dbc.Col(
                                    [dcc.Graph(id="plot4-reservations", figure={})],
                                    width=6,
                                    align="center",
                                ),
                            ]
                        ),
                    ],
                    width=10,
                ),
                dbc.Col(
                    # daq.Thermometer(
                    #     id="eff-commissions-thermometer",
                    #     value=5,
                    #     min=0,
                    #     max=10,
                    #     height=200,
                    #     width=25,
                    #     # style={"margin-bottom": "3000%"},
                    # ),
                    daq.Gauge(
                        id="eff-commissions-gauge",
                        color={
                            "gradient": True,
                            "ranges": {
                                "green": [0, 6],
                                "yellow": [6, 10],
                                "red": [10, 15],
                            },
                        },
                        # value=2,
                        label=dict(
                            label="Effective commissions",
                            style={"font-size": "20px"},
                        ),
                        max=15,
                        min=0,
                        showCurrentValue=True,
                        units="%",
                        style={"color": "black"},
                        className="eff-gauge",
                    ),
                    align="center",
                    width=2,
                ),
            ],
            align="center",
        ),
    ]
)


# Create interactivity between dropdown component and graph
@app.callback(
    Output(component_id="plot1-visits", component_property="figure"),
    Output(component_id="plot2-CPC", component_property="figure"),
    Output(component_id="plot3-margin", component_property="figure"),
    Output(component_id="plot4-reservations", component_property="figure"),
    Output("eff-commissions-gauge", "value"),
    Input("ADR_slider", "value"),
    Input("CPC_slider", "value"),
    Input("Ad_start", "value"),
)
def plot_data(ADR, CPC, Ad_start):
    layout = Layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showline=True,
            showgrid=True,
            gridcolor="rgb(204, 204, 204)",
            showticklabels=True,
            linecolor="rgb(204, 204, 204)",
            linewidth=2,
            ticks="outside",
            griddash="dot",
        ),
        yaxis=dict(
            showline=True,
            showgrid=True,
            gridcolor="rgb(204, 204, 204)",
            showticklabels=True,
            linecolor="rgb(204, 204, 204)",
            linewidth=2,
            ticks="outside",
            griddash="dot",
        ),
        titlefont=dict(family="Font Awesome 6 Free"),
        hovermode="x unified",
    )

    # Additional daily visitors with campaing
    vis = sigmoid(CPC)
    visits_with_campaign = df_visits.values[0].copy()
    visits_with_campaign[:Ad_start] += int(vis)
    cum_visits_with_camp = np.flip(np.cumsum(np.flip(visits_with_campaign)))

    ###################### VISITS
    # plot the visits curve
    fig_visits = go.Figure(layout=layout)
    fig_visits.add_trace(
        go.Scatter(
            x=np.arange(TODAY_LEAD, 365),
            y=cum_visits[TODAY_LEAD:],
            line=dict(color="rgb(21, 76, 121)", width=4),
            fill="tozeroy",
            text="Visits",
            hoverinfo="text",
        )
    )
    fig_visits.add_trace(
        go.Scatter(
            x=np.arange(0, TODAY_LEAD),
            y=cum_visits[:TODAY_LEAD],
            line=dict(color="rgb(21, 76, 121)", width=4, dash="dash"),
            fill="tozeroy",
            text="Visits forecast",
            hoverinfo="text",
        )
    )
    fig_visits.add_trace(
        go.Scatter(
            x=np.arange(0, Ad_start),
            y=cum_visits_with_camp[:Ad_start],
            line=dict(color="rgb(226, 135, 67)", width=4, dash="dash"),
            fill="tonexty",
            text="Visits forecast with ads",
            hoverinfo="text",
            # fill='tozeroy'
        )
    )

    fig_visits.update_layout(
        title="Visits curve",
        xaxis_title="Lead time",
        yaxis_title="Cumulative visits",
        titlefont=dict(size=20),
        title_x=0.5,
    )
    fig_visits.update_yaxes(side="right", range=[0, 1500])
    fig_visits.update_xaxes(range=[200, 0])
    # take out legend
    fig_visits.update_layout(showlegend=False)

    ###################### CPC
    # Create a scatter plot
    fig_CPC = go.Figure(layout=layout)
    fig_CPC.add_trace(
        go.Scatter(
            x=np.arange(0, 1, 0.01),
            y=sigmoid(
                np.arange(0, 1, 0.01)
            ),  # 10 * (1 - np.exp(-10 * np.arange(0, 1, 0.01))),
            line=dict(width=4, color="rgb(21, 76, 121)"),
            text="Additional visits",
            hoverinfo="text",
        )
    )

    fig_CPC.add_trace(go.Scatter(x=[CPC], y=[vis], hoverinfo="none")).update_traces(
        marker_size=12, marker_color="rgb(135, 62, 35)"
    )

    # Add vertical and horizontal dashed lines
    fig_CPC.add_shape(
        type="line",
        x0=0,
        x1=CPC,
        y0=vis,
        y1=vis,
        line=dict(color="rgb(135, 62, 35)", dash="dash"),
        name="",
    )
    fig_CPC.add_shape(
        type="line",
        x0=CPC,
        x1=CPC,
        y0=0,
        y1=vis,
        line=dict(color="rgb(135, 62, 35)", dash="dash"),
        name="",
    )

    # Set axis limits
    fig_CPC.update_xaxes(range=[0, 1])
    fig_CPC.update_yaxes(
        side="right",
        range=[0, 10],
    )

    # take away the legend
    fig_CPC.update_layout(
        title="Additional visits",
        xaxis_title="CPC",
        yaxis_title="Visits",
        showlegend=False,
        titlefont=dict(size=20),
        title_x=0.5,
    )

    ###################### RESERVATIONS
    advance_vect, reservations = get_reservations_from_model(
        ADR=ADR, logreg=logreg, df_visits=df_visits.values
    )
    advance_vect, reservations_campaign = get_reservations_from_model(
        ADR=ADR, logreg=logreg, df_visits=visits_with_campaign
    )
    fig_reservations = go.Figure(layout=layout)
    fig_reservations.add_trace(
        go.Scatter(
            x=advance_vect[TODAY_LEAD:],
            y=reservations[0, TODAY_LEAD:],
            line=dict(color="rgb(21, 76, 121)", width=4),
            fill="tozeroy",
            text="Reservations",
            hoverinfo="text",
        )
    )
    fig_reservations.add_trace(
        go.Scatter(
            x=advance_vect[:TODAY_LEAD],
            y=reservations[0, :TODAY_LEAD],
            line=dict(color="rgb(21, 76, 121)", width=4, dash="dash"),
            fill="tozeroy",
            text="Reservations forecast",
            hoverinfo="text",
        )
    )
    fig_reservations.add_trace(
        go.Scatter(
            x=advance_vect[:Ad_start],
            y=reservations_campaign[0, :Ad_start],
            line=dict(color="rgb(226, 135, 67)", width=4, dash="dash"),
            fill="tonexty",
            text="Reservations forecast with ads",
            hoverinfo="text",
        )
    )
    fig_reservations.update_layout(
        title="Reservations curve",
        xaxis_title="Lead time",
        yaxis_title="Reservations",
        titlefont=dict(size=20),
        title_x=0.5,
    )
    fig_reservations.update_yaxes(
        side="right",
        range=[0, 32],
    )
    fig_reservations.update_xaxes(range=[200, 0])
    # take out legend
    fig_reservations.update_layout(showlegend=False)

    ###################### MARGIN
    fig_margin = go.Figure(layout=layout)

    fig_margin.add_trace(
        go.Scatter(
            x=advance_vect[TODAY_LEAD:],
            y=reservations[0, TODAY_LEAD:] * ADR,
            line=dict(color="rgb(21, 76, 121)", width=4),
            fill="tozeroy",
            text="Margin",
            hoverinfo="text",
        )
    )
    fig_margin.add_trace(
        go.Scatter(
            x=advance_vect[:TODAY_LEAD],
            y=reservations[0, :TODAY_LEAD] * ADR,
            line=dict(color="rgb(21, 76, 121)", width=4, dash="dash"),
            fill="tozeroy",
            text="Margin forecast",
            hoverinfo="text",
        )
    )
    fig_margin.add_trace(
        go.Scatter(
            x=advance_vect[:Ad_start],
            y=reservations_campaign[0, :Ad_start] * ADR
            - CPC * vis * np.arange(Ad_start + 1, 1, -1),
            line=dict(color="rgb(226, 135, 67)", width=4, dash="dash"),
            fill="tonexty",
            text="Margin forecast with ads",
            hoverinfo="text",
        )
    )
    fig_margin.update_layout(
        title="Margin curve",
        xaxis_title="Lead time",
        yaxis_title="Margin",
        titlefont=dict(size=20),
        title_x=0.5,
    )
    # set autoscale on y
    fig_margin.update_yaxes(side="right", range=[0, 7500])
    fig_margin.update_xaxes(range=[200, 0])
    # take out legend
    fig_margin.update_layout(showlegend=False)

    ###################### COMMISSIONS
    # compute the additional reservations
    additional_reservations = reservations_campaign[0, 0] - reservations[0, 0]
    # compute the additional revenue by the campaign
    additional_revenue = additional_reservations * ADR
    # compute the commissions
    commission_percentage = np.clip(
        (vis * CPC * TODAY_LEAD) / (additional_revenue + 0.001) * 100, 0, 15
    )
    # commission_percentage = 12
    return (fig_visits, fig_CPC, fig_margin, fig_reservations, commission_percentage)


# @app.callback(
#     Output("eff-commissions-thermometer", "value"),
#     Output("eff-commissions-thermometer", "color"),
#     Input("CPC_slider", "value"),
# )
# def update_thermometer(value):
#     # update value and color according to colormap 'turbo'
#     color = matplotlib.colormaps["turbo"](value)
#     return value * 10, matplotlib.colors.rgb2hex(color)


# @app.callback(
#     Output("eff-commissions-gauge", "value"),
#     Input("CPC_slider", "value"),
# )
# def update_gauge(value):
#     # update value and color according to colormap 'turbo'
#     # color = matplotlib.colormaps["turbo"](value)
#     return value * 10


if __name__ == "__main__":
    app.run_server(debug=False, port=8002)
