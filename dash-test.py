import datetime

import dash_bootstrap_components as dbc  # pip install dash-bootstrap-components
import dash_daq as daq
import flask
import numpy as np
import pandas as pd  # pip install pandas
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html  # pip install dash
from plotly.graph_objects import Layout

from data_preparation import (
    clip_reservations,
    create_visits_curves,
    get_conversion_rate_model,
    get_reservations_from_model,
    import_data_preprocessed,
)
from margin_optimization import (
    add_visits,
    compset_cutoff,
    compset_cutoff_booking,
    cpc_from_budget,
)

TODAY_LEAD = 75
BOOKING_MULTIPLIER = 1.25
COMMISSION_COSTS = 0.18

GAUGE_SIZE = 130
GAUGE_SIZE_RESPONSIVE = 100

# df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/solar.csv")
app_server = flask.Flask(__name__)  # define flask app.server
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], server=app_server)

app.layout = dbc.Container(
    [
        html.H1(
            "DEMO CAMPAIGNS",
            className="mb-2",
            style={"textAlign": "center"},
        ),
        dbc.Col(
            [
                dbc.Row(
                    dcc.RadioItems(
                        id="t_stay",
                        options=[
                            {"label": "Low Season", "value": "2023-06-30"},
                            {"label": "Middle Season", "value": "2023-07-28"},
                            {"label": "High Season", "value": "2023-08-15"},
                        ],
                        value="2023-06-30",
                        inline=True,
                        style={
                            "textAlign": "center",
                            "margin": "15px",
                            "font-size": "25px",
                            "font-family": "Source Sans 3",
                        },
                    )
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
                                    style={
                                        "textAlign": "center",
                                    },
                                ),
                                # slider of the ADR
                                dcc.Slider(
                                    id="ADR_slider",
                                    min=50,
                                    max=350,
                                    step=1,
                                    value=180,
                                    # marks=slider_marks_ADR,
                                    tooltip={
                                        "placement": "top",
                                        "always_visible": True,
                                    },
                                ),
                            ],
                            width={"size": 3, "offset": 0},
                        ),
                        # slider of the CPC
                        dbc.Col(
                            [
                                html.H3(
                                    "Total Budget",
                                    className="mb-2",
                                    style={"textAlign": "center"},
                                ),
                                # slider of the ADR
                                dcc.Slider(
                                    id="budget_slider",
                                    min=0,
                                    max=100,
                                    step=5,
                                    value=25,
                                    # marks={i: f"{i}€" for i in range(0, 101, 20)},
                                    tooltip={"placement": "top", "always_visible": True}
                                    # handleLabel={"showCurrentValue": True, "label": "VALUE"},
                                ),
                            ],
                            width={"size": 3, "offset": 0},
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
                                    tooltip={
                                        "placement": "top",
                                        "always_visible": True,
                                    },
                                    # marks={i: str(i) for i in range(0, 76, 10)},
                                ),
                            ],
                            width={"size": 3, "offset": 0},
                        ),
                        dbc.Col(
                            [
                                html.H3(
                                    "Booking.com markup",
                                    className="mb-2",
                                    style={"textAlign": "center"},
                                ),
                                # slider of the ADR
                                dcc.Slider(
                                    id="Booking_markup",
                                    min=0,
                                    max=25,
                                    step=1,
                                    value=15,
                                    tooltip={
                                        "placement": "top",
                                        "always_visible": True,
                                    },
                                    # marks={i: str(i) for i in range(0, 26, 5)},
                                ),
                            ],
                            width={"size": 3, "offset": 0},
                        ),
                    ],
                ),
            ],
            class_name="controllers",
            width=12,
            style={
                "background-color": "#fff",
                "border-radius": "15px",
                # "margin": "5px",
            },
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            id="plot1-visits",
                                            figure={},
                                        )
                                    ],
                                    width=6,
                                    align="center",
                                    style={"padding": "0px"},
                                ),
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            id="plot5-booking",
                                            figure={},
                                        )
                                    ],
                                    width=6,
                                    align="center",
                                    style={"padding": "0px"},
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            id="plot3-margin",
                                            figure={},
                                        )
                                    ],
                                    width=6,
                                    align="center",
                                    style={"padding": "0px"},
                                ),
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            id="plot4-reservations",
                                            figure={},
                                        )
                                    ],
                                    width=6,
                                    align="center",
                                    style={"padding": "0px"},
                                ),
                            ]
                        ),
                    ],
                    width=10,
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                html.Div(
                                    [
                                        html.Div(className="tab-revenue-box"),
                                        html.H2(
                                            "Expected Revenue",
                                            className="total-revenue-title",
                                        ),
                                        html.H2(id="total-revenue"),
                                    ],
                                    style={
                                        "background-color": "#fff",
                                        "border-radius": "10px",
                                    },
                                    className="total-revenue-box",
                                ),
                                html.Div(
                                    [
                                        html.Div(className="tab-margin-box"),
                                        html.H2(
                                            "Expected Margin",
                                            className="total-margin-title",
                                        ),
                                        html.H2(id="total-margin"),
                                    ],
                                    className="total-margin-box",
                                ),
                                html.Div(
                                    [
                                        html.Div(className="tab-comm-box"),
                                        html.H2(
                                            "Effective commissions",
                                            className="eff-comm-title",
                                        ),
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
                                            # label=dict(
                                            #     label="Effective commissions",
                                            #     style={"font-size": "18px"},
                                            # ),
                                            max=15,
                                            min=0,
                                            showCurrentValue=True,
                                            units="%",
                                            className="eff-gauge",
                                            size=GAUGE_SIZE,
                                        ),
                                    ],
                                    className="eff-comm-box",
                                ),
                                # daq.Gauge(
                                #     id="eff-commissions-gauge",
                                #     color={
                                #         "gradient": True,
                                #         "ranges": {
                                #             "green": [0, 6],
                                #             "yellow": [6, 10],
                                #             "red": [10, 15],
                                #         },
                                #     },
                                #     # value=2,
                                #     label=dict(
                                #         label="Effective commissions",
                                #         style={"font-size": "18px"},
                                #     ),
                                #     max=15,
                                #     min=0,
                                #     showCurrentValue=True,
                                #     units="%",
                                #     className="eff-gauge",
                                #     size=180,
                                # ),
                            ]
                        ),
                    ],
                    align="start",
                    width=2,
                ),
            ],
            align="center",
        ),
        # dbc.Row(
        #     dbc.Col(
        #         [dcc.Graph(id="plot2-CPC", figure={})],
        #         align="center",
        #         width={"size": 5, "offset": 3},
        #     ),
        # ),
        # dcc.Store stores the intermediate value
        dcc.Store(id="df_visits_t_stay"),
    ]
)

# import pre-processed data
df_t_stay = import_data_preprocessed()
CR_model = get_conversion_rate_model(df_t_stay)

map_t_stay_cluster = {"2023-07-28": 2, "2023-06-30": 1, "2023-08-15": 3}
map_compset_price = {"2023-07-28": 220, "2023-06-30": 150, "2023-08-15": 250}

optimal_values = {
    "2023-07-28": [191, 100, 75, 4],
    "2023-06-30": [129, 100, 75, 0],
    "2023-08-15": [235, 100, 75, 2],
}


# Create interactivity between dropdown component and graph
@app.callback(
    Output("df_visits_t_stay", "data"),
    Output("ADR_slider", "marks"),
    Output("budget_slider", "marks"),
    Output("Ad_start", "marks"),
    Output("Booking_markup", "marks"),
    Input("t_stay", "value"),
)
def create_model(t_stay):
    # select the visits for the t_stay date
    t_stay_date = datetime.datetime.strptime(t_stay, "%Y-%m-%d").date()
    df_visits = create_visits_curves(df_t_stay[df_t_stay.T_stay.dt.date == t_stay_date])

    # ADR slider marks
    ADR_marks = {i: f"{i}€" for i in range(50, 351, 50)}
    # add the compset price in rgba(142, 202, 230,1) and above slider
    ADR_marks[map_compset_price[t_stay]] = {
        "label": "compset",
        "style": {"color": "rgb(222, 49, 99)", "top": "10px", "font-weight": "bold"},
    }
    ADR_marks[optimal_values[t_stay][0]] = {
        "label": "opt",
        "style": {
            "color": "rgb(64, 224, 208)",
            "bottom": "30px",
            "font-weight": "bold",
        },
    }

    # budget slider marks
    budget_marks = {i: f"{i}€" for i in range(0, 101, 20)}
    budget_marks[optimal_values[t_stay][1]] = {
        "label": "opt",
        "style": {
            "color": "rgb(64, 224, 208)",
            "bottom": "30px",
            "font-weight": "bold",
        },
    }

    # Ad start slider marks
    Ad_start_marks = {i: str(i) for i in range(0, 76, 10)}
    Ad_start_marks[optimal_values[t_stay][2]] = {
        "label": "opt",
        "style": {
            "color": "rgb(64, 224, 208)",
            "bottom": "30px",
            "font-weight": "bold",
        },
    }

    # Booking markup slider marks
    Booking_markup_marks = {i: str(i) for i in range(0, 26, 5)}
    Booking_markup_marks[optimal_values[t_stay][3]] = {
        "label": "opt",
        "style": {
            "color": "rgb(64, 224, 208)",
            "bottom": "30px",
            "font-weight": "bold",
        },
    }

    # store in
    return (
        df_visits.to_json(date_format="iso", orient="split"),
        ADR_marks,
        budget_marks,
        Ad_start_marks,
        Booking_markup_marks,
    )


# t_stay = "2023-06-30"


# Create interactivity between dropdown component and graph
@app.callback(
    Output(component_id="plot1-visits", component_property="figure"),
    Output(component_id="plot5-booking", component_property="figure"),
    Output(component_id="plot3-margin", component_property="figure"),
    Output(component_id="plot4-reservations", component_property="figure"),
    # Output(component_id="plot2-CPC", component_property="figure"),
    Output("eff-commissions-gauge", "value"),
    Output("total-margin", "children"),
    Output("total-revenue", "children"),
    Input("ADR_slider", "value"),
    Input("budget_slider", "value"),
    Input("Ad_start", "value"),
    Input("Booking_markup", "value"),
    Input("df_visits_t_stay", "data"),
    Input("t_stay", "value"),
)
def plot_data(ADR, budget, Ad_start, Booking_markup, df_visits_t_stay, t_stay):
    layout = Layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showline=True,
            showgrid=False,
            # gridcolor="rgb(204, 204, 204)",
            showticklabels=True,
            linecolor="rgb(204, 204, 204)",
            linewidth=2,
            ticks="outside",
            # griddash="dot",
        ),
        yaxis=dict(
            showline=True,
            showgrid=False,
            # gridcolor="rgb(204, 204, 204)",
            showticklabels=True,
            linecolor="rgb(204, 204, 204)",
            linewidth=2,
            ticks="outside",
            # griddash="dot",
        ),
        titlefont=dict(family="Source Sans 3"),
        font=dict(family="Source Sans 3"),
        hovermode="x unified",
    )
    cluster = map_t_stay_cluster[t_stay]
    compset_price = map_compset_price[t_stay]
    df_visits = pd.read_json(df_visits_t_stay, orient="split")
    cum_visits = np.flip(np.cumsum(np.flip(df_visits.values)))
    visits_booking = df_visits.values * BOOKING_MULTIPLIER
    cum_visits_booking = np.flip(np.cumsum(np.flip(visits_booking)))

    CPC = cpc_from_budget(budget / Ad_start)

    # Additional daily visitors with campaing
    vis = add_visits(CPC)
    visits_with_campaign = np.zeros((365))
    visits_with_campaign[:Ad_start] += vis
    cum_visits_with_camp = np.flip(
        np.cumsum(np.flip(df_visits.values[0] + visits_with_campaign))
    )

    ###################### VISITS
    # plot the visits curve
    fig_visits = go.Figure(layout=layout)
    fig_visits.add_trace(
        go.Scatter(
            x=np.arange(TODAY_LEAD, 365),
            y=cum_visits[TODAY_LEAD:],
            line=dict(color="rgba(1, 99, 155,1)", width=4),
            fill="tozeroy",
            fillcolor="rgba(1, 99, 155,1)",
            text="Visits",
            hoverinfo="text",
        )
    )
    fig_visits.add_trace(
        go.Scatter(
            x=np.arange(Ad_start, TODAY_LEAD + 1),
            y=cum_visits[Ad_start : TODAY_LEAD + 1],
            line=dict(color="rgba(1, 99, 155,0.5)", width=4),
            fill="tozeroy",
            fillcolor="rgba(1, 99, 155,0.5)",
            text="Visits forecast",
            hoverinfo="text",
            mode="lines",
        )
    )
    fig_visits.add_trace(
        go.Scatter(
            x=np.arange(0, Ad_start + 1),
            y=cum_visits[: Ad_start + 1],
            line=dict(color="rgba(1, 99, 155,0.5)", width=4),
            fill="tozeroy",
            fillcolor="rgba(1, 99, 155,0.5)",
            text="Visits forecast",
            hoverinfo="text",
            mode="lines",
        )
    )

    if budget > 0 and Ad_start > 0:
        fig_visits.add_trace(
            go.Scatter(
                x=np.arange(0, Ad_start + 1),
                y=cum_visits_with_camp[: Ad_start + 1],
                line=dict(color="rgba(243, 122, 0,0.5)", width=4),
                fill="tonexty",
                fillcolor="rgba(243, 122, 0,0.5)",
                text="Visits forecast with ads",
                hoverinfo="text",
                mode="lines"
                # fill='tozeroy'
            )
        )

    fig_visits.update_layout(
        title="Visits curve",
        xaxis_title="Lead time [d]",
        yaxis_title="Cumulative visits",
        titlefont=dict(size=20),
        title_x=0.5,
    )
    xticks = list(np.arange(0, 201, 50))
    xlabels = [str(i) for i in xticks]
    xticks.append(TODAY_LEAD)
    xlabels.append("Today")
    fig_visits.update_yaxes(side="right", range=[0, 1500])
    fig_visits.update_xaxes(range=[200, 0], tickvals=xticks, ticktext=xlabels)
    # take out legend
    fig_visits.update_layout(showlegend=False)

    ###################### VISITS ON BOOKING
    # plot the visits curve on booking
    fig_booking = go.Figure(layout=layout)
    fig_booking.add_trace(
        go.Scatter(
            x=np.arange(TODAY_LEAD, 365),
            y=cum_visits_booking[TODAY_LEAD:],
            line=dict(color="rgba(142, 202, 230,1)", width=4),
            fill="tozeroy",
            fillcolor="rgba(142, 202, 230,1)",
            text="Visits on Booking.com",
            hoverinfo="text",
        )
    )
    fig_booking.add_trace(
        go.Scatter(
            x=np.arange(0, TODAY_LEAD + 1),
            y=cum_visits_booking[: TODAY_LEAD + 1],
            line=dict(color="rgba(142, 202, 230,0.5)", width=4),
            fill="tozeroy",
            fillcolor="rgba(142, 202, 230,0.5)",
            text="Visits forecast on Booking.com",
            hoverinfo="text",
        )
    )

    fig_booking.update_layout(
        title="Visits curve on Booking.com",
        xaxis_title="Lead time [d]",
        yaxis_title="Cumulative visits",
        titlefont=dict(size=20),
        title_x=0.5,
    )

    fig_booking.update_xaxes(range=[200, 0], tickvals=xticks, ticktext=xlabels)
    fig_booking.update_yaxes(side="right", range=[0, 2000])
    # take out legend
    fig_booking.update_layout(showlegend=False)

    ###################### CPC
    # Create a scatter plot
    fig_CPC = go.Figure(layout=layout)
    fig_CPC.add_trace(
        go.Scatter(
            x=np.arange(0, 5, 0.01),
            y=add_visits(np.arange(0, 5, 0.01)),
            line=dict(width=4, color="rgba(1, 99, 155,1)"),
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
        x0=CPC,
        x1=5,
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
    fig_CPC.update_xaxes(range=[-0.05, 5])
    fig_CPC.update_yaxes(
        side="right",
        range=[0, 12],
    )

    # take away the legend
    fig_CPC.update_layout(
        title="Additional visits",
        xaxis_title="CPC [euro]",
        yaxis_title="Visits",
        showlegend=False,
        titlefont=dict(size=20),
        title_x=0.5,
    )

    ###################### RESERVATIONS
    advance_vect, reservations = get_reservations_from_model(
        ADR=ADR,
        logreg=CR_model[cluster],
        df_visits=df_visits.values,
        compset_cutoff=compset_cutoff(ADR, compset_price),
    )
    _, reservations_campaign = get_reservations_from_model(
        ADR=ADR,
        logreg=CR_model[cluster],
        df_visits=visits_with_campaign,
        compset_cutoff=compset_cutoff(ADR, compset_price),
    )
    _, reservations_booking = get_reservations_from_model(
        ADR=ADR * (1 + Booking_markup / 100),
        logreg=CR_model[cluster],
        df_visits=visits_booking,
        compset_cutoff=compset_cutoff_booking(
            ADR * (1 + Booking_markup / 100), compset_price
        ),
    )

    # clip the reservations
    [reservations, reservations_campaign, reservations_booking] = clip_reservations(
        [reservations, reservations_campaign, reservations_booking], 32
    )
    fig_reservations = go.Figure(layout=layout)

    # reservations on direct
    fig_reservations.add_trace(
        go.Scatter(
            x=advance_vect[TODAY_LEAD:],
            y=reservations[0, TODAY_LEAD:],
            line=dict(color="rgba(1, 99, 155,1)", width=4),
            fill="tozeroy",
            fillcolor="rgba(1, 99, 155,1)",
            text="Reservations on direct",
            hoverinfo="text",
        )
    )

    # reservations on booking plus direct up to today
    fig_reservations.add_trace(
        go.Scatter(
            x=advance_vect[TODAY_LEAD:],
            y=reservations_booking[0, TODAY_LEAD:] + reservations[0, TODAY_LEAD:],
            line=dict(color="rgba(142, 202, 230,1)", width=4),
            fill="tonexty",
            fillcolor="rgba(142, 202, 230,1)",
            text="Reservations on Booking.com",
            hoverinfo="text",
        )
    )
    # reservations on direct forecast
    fig_reservations.add_trace(
        go.Scatter(
            x=advance_vect[Ad_start : TODAY_LEAD + 1],
            y=reservations[0, Ad_start : TODAY_LEAD + 1],
            line=dict(color="rgba(1, 99, 155,0.5)", width=4),
            fill="tozeroy",
            fillcolor="rgba(1, 99, 155,0.5)",
            text="Reservations forecast on direct",
            hoverinfo="text",
            mode="lines",
        )
    )

    # reservations on booking + direct forecast
    fig_reservations.add_trace(
        go.Scatter(
            x=advance_vect[Ad_start : TODAY_LEAD + 1],
            y=reservations_booking[0, Ad_start : TODAY_LEAD + 1]
            + reservations[0, Ad_start : TODAY_LEAD + 1],
            line=dict(color="rgba(142, 202, 230,0.5)", width=4),
            fill="tonexty",
            fillcolor="rgba(142, 202, 230,0.5)",
            text="Reservations forecast on Booking.com",
            hoverinfo="text",
            mode="lines",
        )
    )

    # reservations on direct forecast
    fig_reservations.add_trace(
        go.Scatter(
            x=advance_vect[: Ad_start + 1],
            y=reservations[0, : Ad_start + 1],
            line=dict(color="rgba(1, 99, 155,0.5)", width=4),
            fill="tozeroy",
            fillcolor="rgba(1, 99, 155,0.5)",
            text="Reservations forecast on direct",
            hoverinfo="text",
            mode="lines",
        )
    )

    # reservations on booking + direct forecast
    fig_reservations.add_trace(
        go.Scatter(
            x=advance_vect[: Ad_start + 1],
            y=reservations_booking[0, : Ad_start + 1] + reservations[0, : Ad_start + 1],
            line=dict(color="rgba(142, 202, 230,0.5)", width=4),
            fill="tonexty",
            fillcolor="rgba(142, 202, 230,0.5)",
            text="Reservations forecast on Booking.com",
            hoverinfo="text",
            mode="lines",
        )
    )

    # reservations on booking plus direct forecast with ads
    if budget > 0 and Ad_start > 0:
        fig_reservations.add_trace(
            go.Scatter(
                x=advance_vect[: Ad_start + 1],
                y=reservations_campaign[0, : Ad_start + 1]
                + reservations[0, : Ad_start + 1]
                + reservations_booking[0, : Ad_start + 1],
                line=dict(color="rgba(243, 122, 0,0.5)", width=4),
                fill="tonexty",
                fillcolor="rgba(243, 122, 0,0.5)",
                text="Reservations forecast with ads",
                hoverinfo="text",
                mode="lines",
            )
        )
    fig_reservations.update_layout(
        title="Reservations curve",
        xaxis_title="Lead time [d]",
        yaxis_title="Reservations",
        titlefont=dict(size=20),
        title_x=0.5,
    )
    fig_reservations.update_yaxes(
        side="right",
        range=[0, 35],
    )

    fig_reservations.update_xaxes(range=[200, 0], tickvals=xticks, ticktext=xlabels)
    # take out legend
    fig_reservations.update_layout(showlegend=False)
    ###################### MARGIN
    fig_margin = go.Figure(layout=layout)
    # margin on booking up to today
    revenue_booking = reservations_booking * (ADR + Booking_markup)
    margin_booking = revenue_booking * (1 - COMMISSION_COSTS)
    direct_revenue = reservations * ADR

    # margin on direct up to today
    fig_margin.add_trace(
        go.Scatter(
            x=advance_vect[TODAY_LEAD:],
            y=direct_revenue[0, TODAY_LEAD:],
            line=dict(color="rgba(1, 99, 155,1)", width=4),
            fill="tozeroy",
            fillcolor="rgba(1, 99, 155,1)",
            text="Margin direct",
            hoverinfo="text",
            mode="lines",
        )
    )
    # margin on booking plus direct up to today
    fig_margin.add_trace(
        go.Scatter(
            x=advance_vect[TODAY_LEAD:],
            y=margin_booking[0, TODAY_LEAD:] + margin_booking[0, TODAY_LEAD:],
            line=dict(color="rgba(142, 202, 230,1)", width=4),
            fill="tonexty",
            fillcolor="rgba(142, 202, 230,1)",
            text="Margin on Booking.com",
            hoverinfo="text",
            mode="lines",
        )
    )

    # forecasting on direct
    fig_margin.add_trace(
        go.Scatter(
            x=advance_vect[Ad_start : TODAY_LEAD + 1],
            y=direct_revenue[0, Ad_start : TODAY_LEAD + 1],
            line=dict(color="rgba(1, 99, 155,0.5)", width=4),
            fill="tozeroy",
            fillcolor="rgba(1, 99, 155,0.5)",
            text="Margin forecast direct",
            hoverinfo="text",
            mode="lines",
        )
    )

    # forecasting on direct plus booking
    fig_margin.add_trace(
        go.Scatter(
            x=advance_vect[Ad_start : TODAY_LEAD + 1],
            y=margin_booking[0, Ad_start : TODAY_LEAD + 1]
            + direct_revenue[0, Ad_start : TODAY_LEAD + 1],
            line=dict(color="rgba(142, 202, 230,0.5)", width=4),
            fill="tonexty",
            fillcolor="rgba(142, 202, 230,0.5)",
            text="Margin forecast Booking.com",
            hoverinfo="text",
            mode="lines",
        )
    )

    # forecasting on direct
    fig_margin.add_trace(
        go.Scatter(
            x=advance_vect[: Ad_start + 1],
            y=direct_revenue[0, : Ad_start + 1],
            line=dict(color="rgba(1, 99, 155,0.5)", width=4),
            fill="tozeroy",
            fillcolor="rgba(1, 99, 155,0.5)",
            text="Margin forecast direct",
            hoverinfo="text",
            mode="lines",
        )
    )

    # forecasting on direct plus booking
    fig_margin.add_trace(
        go.Scatter(
            x=advance_vect[: Ad_start + 1],
            y=margin_booking[0, : Ad_start + 1] + direct_revenue[0, : Ad_start + 1],
            line=dict(color="rgba(142, 202, 230,0.5)", width=4),
            fill="tonexty",
            fillcolor="rgba(142, 202, 230,0.5)",
            text="Margin forecast Booking.com",
            hoverinfo="text",
            mode="lines",
        )
    )

    # forecasting on direct plus booking plus ads
    if budget > 0 and Ad_start > 0:
        fig_margin.add_trace(
            go.Scatter(
                x=advance_vect[:Ad_start],
                y=reservations_campaign[0, :Ad_start] * ADR
                + margin_booking[0, :Ad_start]
                + direct_revenue[0, :Ad_start]
                - CPC * vis * np.arange(Ad_start + 1, 1, -1),
                line=dict(color="rgba(243, 122, 0,0.5)", width=4),
                fill="tonexty",
                fillcolor="rgba(243, 122, 0,0.5)",
                text="Margin forecast of ads",
                hoverinfo="text",
                mode="lines",
            )
        )
    fig_margin.update_layout(
        title="Margin curve",
        xaxis_title="Lead time [d]",
        yaxis_title="Margin [euro]",
        titlefont=dict(size=20),
        title_x=0.5,
    )
    # set autoscale on y
    fig_margin.update_yaxes(side="right", range=[0, 10000])
    fig_margin.update_xaxes(range=[200, 0], tickvals=xticks, ticktext=xlabels)
    # take out legend
    fig_margin.update_layout(showlegend=False)

    ###################### TOTAL MARGIN
    total_revenue = (
        reservations_campaign[0, 0] * ADR + revenue_booking[0, 0] + direct_revenue[0, 0]
    )
    total_margin = (
        reservations_campaign[0, 0] * ADR
        + margin_booking[0, 0]
        + direct_revenue[0, 0]
        - CPC * vis * (Ad_start + 1)
    )

    ###################### COMMISSIONS
    # compute the additional reservations
    additional_reservations = reservations_campaign[0, 0]
    # compute the additional revenue by the campaign
    additional_revenue = additional_reservations * ADR
    # compute the commissions
    commission_percentage = np.clip(
        (CPC * vis * (Ad_start + 1)) / (additional_revenue + 0.001) * 100, 0, 15
    )

    # commission_percentage = 12
    return (
        fig_visits,
        fig_booking,
        fig_margin,
        fig_reservations,
        # fig_CPC,
        commission_percentage,
        f"{total_margin:.2f}€",
        f"{total_revenue:.2f}€",
    )


if __name__ == "__main__":
    app.run_server(debug=False, port=8002)
