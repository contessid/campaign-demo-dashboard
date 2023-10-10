import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data_preparation import (
    clip_reservations,
    create_visits_curves,
    get_conversion_rate_model,
    import_data_preprocessed,
)

EXPONENT = 1 / 2
ADD_VISITS_COEFF = 5
COMMISSION_COSTS = 0.18
BOOKING_MULTIPLIER = 1.25
# COMPSET_PRICE = 220
COMPSET_SENSITIVITY = 0.01


def compset_cutoff(adr, COMPSET_PRICE):
    # sigmoid function that cuts the conversion rate
    # when the ADR is above the compset price
    return 1 / (1 + np.exp(-COMPSET_SENSITIVITY * (COMPSET_PRICE + 5 - adr)))


def compset_cutoff_booking(adr, COMPSET_PRICE):
    # sigmoid function that cuts the conversion rate
    # when the ADR is above the compset price
    return 1 / (1 + np.exp(-COMPSET_SENSITIVITY * ((COMPSET_PRICE + 5) * 1.15 - adr)))


def add_visits(CPC):
    # return 10 * (1 - np.exp(-5 * CPC))
    return ADD_VISITS_COEFF * (np.power(CPC, EXPONENT))


def cpc_from_budget(budget):
    return np.power(budget / ADD_VISITS_COEFF, 1 / (EXPONENT + 1))


def add_daily_visits_from_budget(budget):
    return add_visits(cpc_from_budget(budget))


def cost_function_campaign(inputs, visits, CR_model, BOOKING_MULTIPLIER, compset_price):
    # variabili
    budget = inputs[0]
    adr = inputs[1]
    advance_campaign = int(inputs[2])
    markup = inputs[3]

    margin = compute_total_margin(
        visits,
        CR_model,
        BOOKING_MULTIPLIER,
        compset_price,
        adr,
        budget,
        advance_campaign,
        markup,
    )

    return -margin


def compute_total_margin(
    visits,
    CR_model,
    BOOKING_MULTIPLIER,
    compset_price,
    adr,
    budget,
    advance_campaign,
    markup,
):
    N_visits_ads = add_daily_visits_from_budget(budget / advance_campaign)
    # timing della campagna
    campaing_timing = np.zeros((365,))
    campaing_timing[0 : int(advance_campaign)] = 1

    ############ DIRECT
    # define the grid on which compute the CR
    advance_vect = np.arange(1, 366)
    grid = np.zeros((365, 2))
    grid[:, 0] = adr  # ADR
    grid[:, 1] = advance_vect  # Advance

    # compute conversion rate
    CR = CR_model.predict_proba(grid)[:, 1]
    CR = np.reshape(CR, (1, 365))
    ############ BOOKING.COM
    booking_price = adr * (1 + markup / 100)
    grid = np.zeros((365, 2))
    grid[:, 0] = booking_price  # ADR
    grid[:, 1] = advance_vect  # Advance

    # compute conversion rate
    CR_booking = CR_model.predict_proba(grid)[:, 1]
    CR_booking = np.reshape(CR_booking, (1, 365))
    # compute reconstructed reservations
    reserv_direct = visits * CR * compset_cutoff(adr, compset_price)
    reserv_direct = np.flip(np.cumsum(np.flip(reserv_direct), axis=1))
    reserv_booking = (
        visits
        * BOOKING_MULTIPLIER
        * CR_booking
        * compset_cutoff_booking(booking_price, compset_price)
    )
    reserv_booking = np.flip(np.cumsum(np.flip(reserv_booking), axis=1))
    reserv_ads = (
        (N_visits_ads * campaing_timing) * CR * compset_cutoff(adr, compset_price)
    )
    reserv_ads = np.flip(np.cumsum(np.flip(reserv_ads), axis=1))
    reserv_direct, reserv_ads, reserv_booking = clip_reservations(
        [reserv_direct, reserv_ads, reserv_booking], 32
    )

    revenue_day_with_ads = (
        reserv_direct[0, 0] * adr
        + reserv_booking[0, 0] * (adr + markup) * (1 - COMMISSION_COSTS)
        + reserv_ads[0, 0] * adr
        - budget
    )

    return revenue_day_with_ads


def get_optimal_margin(visits, CR_model, BOOKING_MULTIPLIER, compset_price):
    bounds = [(0, 100), (50, 350), (1, 75), (0, 25)]

    fun_val = []
    opt_par = []

    for i in range(30):
        initial_guess = np.array(
            [
                np.random.randint(50 + 1),
                np.random.randint(200 + 1) + 100,
                np.random.randint(50 + 1),
                np.random.randint(10 + 1),
            ]
        )  # cpc, adr, advance, markup
        result = minimize(
            cost_function_campaign,
            initial_guess,
            method="Nelder-Mead",
            bounds=bounds,
            args=(visits, CR_model, BOOKING_MULTIPLIER, compset_price),
        )

        # Extract the optimized parameters
        opt_par.append(result.x)
        fun_val.append(result.fun)

    fun_val = np.array(fun_val)
    best = np.argmin(fun_val)

    print("optimal parameters:", opt_par[best])
    print("optimal margin:", -fun_val[best])
    print("----------------------------")


if __name__ == "__main__":
    t_stay = "2023-06-30"
    map_t_stay_cluster = {"2023-07-28": 2, "2023-06-30": 1, "2023-08-15": 3}
    compset_price = {"2023-07-28": 220, "2023-06-30": 150, "2023-08-15": 250}
    cluster = map_t_stay_cluster[t_stay]

    # load data and model
    df_t_stay = import_data_preprocessed()
    CR_model = get_conversion_rate_model(df_t_stay)

    t_stay_date = datetime.datetime.strptime(t_stay, "%Y-%m-%d").date()
    df_visits = create_visits_curves(df_t_stay[df_t_stay.T_stay.dt.date == t_stay_date])

    # optimal margin
    get_optimal_margin(
        df_visits.values[0],
        CR_model[cluster],
        BOOKING_MULTIPLIER,
        compset_price[t_stay],
    )
    # compute margin
    # budget = 25
    # adr = 180
    # advance_campaign = 60
    # markup = 15
    # margin = compute_total_margin(
    #     df_visits.values[0],
    #     CR_model[cluster],
    #     BOOKING_MULTIPLIER,
    #     adr,
    #     budget,
    #     advance_campaign,
    #     markup,
    # )
    # print("margin:", margin)
