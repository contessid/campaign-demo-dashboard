import datetime
import json
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REQUESTS_PATH = "./data/requests.csv"
CLUSTER_PATH = "./data/clusters.pkl"
REQUESTS_PREPROCESSED_PATH = "./data/requests_t_stay.csv"
CLUSTER = 2
ARRIVAL_DATE = datetime.date(2023, 7, 28)


def booking_attempts_t_res_to_t_stay(booking_attempts_df):
    """Transforms every request in N requests of 1 day where N is the LOS"""
    new_att_list = []
    for i in range(len(booking_attempts_df)):
        att = booking_attempts_df.iloc[i].to_dict()
        # append the first day of stay
        new_att_list.append(att)
        start_stay = att["T_res"]
        LOS = att["LOS"]

        # transform single reservations for multiple days in multiple reservations for single day
        for k in range(1, LOS, 1):
            new_t_res = start_stay + datetime.timedelta(days=k)
            new_att = att.copy()
            new_att["T_res"] = new_t_res
            new_att["Advance"] += k
            new_att_list.append(new_att)
    df_new_bkgs = pd.DataFrame(data=new_att_list)
    # df_new_bkgs.drop(columns=['T_out','LOS'],inplace=True)
    df_new_bkgs.drop(columns=["LOS"], inplace=True)
    df_new_bkgs.rename(columns={"T_res": "T_stay"}, inplace=True)
    return df_new_bkgs


def parse_dict(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}


def import_requests_data():
    df_requests = pd.read_csv(
        REQUESTS_PATH,
        parse_dates=["T_res", "T_book", "T_book_day"],
        converters={"AvailableRoomRates": parse_dict, "SelectedRoomRates": parse_dict},
    )
    return df_requests


def import_clusters_data():
    with open(CLUSTER_PATH, "rb") as file:
        # Load the data from the pkl file
        return pickle.load(file)


def import_data():
    df_requests = import_requests_data()
    df_t_stay = booking_attempts_t_res_to_t_stay(df_requests)
    clusters = import_clusters_data()
    # assign cluster to the dates
    df_t_stay["Cluster"] = 0
    for i in range(len(df_t_stay)):
        for cluster, val in enumerate(clusters):
            if (
                datetime.datetime.strptime(
                    str(df_t_stay.loc[i, "T_stay"]), "%Y-%m-%d %H:%M:%S"
                ).date()
                in clusters[cluster]["dates"]
            ):
                df_t_stay.loc[i, "Cluster"] = cluster
    return df_t_stay


def import_data_preprocessed():
    df_t_stay = pd.read_csv(
        REQUESTS_PREPROCESSED_PATH,
        parse_dates=["T_book", "T_stay", "T_book_day"],
        converters={"AvailableRoomRates": parse_dict, "SelectedRoomRates": parse_dict},
    )

    return df_t_stay


def logistic_fit(X, y):
    # split in train and test dataset

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=1
    )

    # import the class

    # instantiate the model (using the default parameters)
    # logreg = LogisticRegression(random_state=16, max_iter=1000,penalty='l1', solver='liblinear',tol=1e-8)
    logreg = LogisticRegression(
        random_state=16, max_iter=1000, penalty="l2", solver="newton-cholesky", tol=1e-8
    )

    # fit the model with data
    logreg.fit(X_train, y_train)

    # prediction is the probability of being accepted
    y_pred_test = logreg.predict_proba(X_test)[:, 1]
    y_pred_train = logreg.predict_proba(X_train)[:, 1]

    print("residuals on train:", np.sum((y_train - y_pred_train) ** 2) / len(y_train))
    print("residuals on test:", np.sum((y_test - y_pred_test) ** 2) / len(y_test))

    return logreg


def create_visits_curves(my_df):
    """Create the number of visits as function of advance curve for each T_stay"""
    T_stay_list = np.unique(my_df.T_stay)

    BOOKING_WINDOW = 365
    # create the matrix of the number of visits: each row is a date and the columns are the advance values
    visits_curves = np.zeros((T_stay_list.shape[0], BOOKING_WINDOW))
    index_t_stay = 0

    for t_stay in T_stay_list:
        # retrieve the visits for the t_stay
        visits_per_day = my_df.loc[my_df.T_stay == t_stay]
        # count the visits for every t_stay and creates a dict whose keys are
        # the advance values and values are number of entries for that advance
        n_visits_per_advance = Counter(visits_per_day["Advance"])
        # from the dictionary, fill the martix
        for idx, val in n_visits_per_advance.items():
            if idx < 365:
                visits_curves[index_t_stay, idx] = val

        index_t_stay += 1

    df_visits = pd.DataFrame(
        data=visits_curves.astype(int),
        columns=[i for i in range(BOOKING_WINDOW)],
        index=T_stay_list,
    )
    df_visits.index.name = "T_stay"

    return df_visits


def create_incremental_booking_curves(df_new_bkgs):
    ### create the booking curves
    T_res_list = np.unique(df_new_bkgs.T_stay)
    BOOKING_WINDOW = 250
    booking_curves = np.zeros((T_res_list.shape[0], BOOKING_WINDOW))
    index_t_res = 0
    for t_res in T_res_list:
        # retrieve the bookings for the t_res
        bookings = df_new_bkgs[df_new_bkgs.T_stay == t_res]
        for adv in bookings.Advance:
            if adv >= BOOKING_WINDOW:
                booking_curves[index_t_res, BOOKING_WINDOW - 1] += 1
            else:
                booking_curves[index_t_res, adv] += 1
        index_t_res += 1

    # incremental booking curves
    booking_curves = np.flip(np.cumsum(np.flip(booking_curves, axis=1), axis=1), axis=1)

    data_bkg_curves = pd.DataFrame(
        data=booking_curves.astype(int),
        columns=[i for i in range(BOOKING_WINDOW)],
        index=T_res_list,
    )
    data_bkg_curves.index.name = "DATE"
    return data_bkg_curves


def logistic_conversion_rate(df_t_stay):
    input_vars = ["ADR", "Advance"]
    X = (
        df_t_stay[df_t_stay.Cluster == CLUSTER][input_vars]
        .reset_index(drop=True)
        .values
    )
    y = (
        df_t_stay[df_t_stay.Cluster == CLUSTER]["Accepted"]
        .reset_index(drop=True)
        .values
    )
    logreg = logistic_fit(X, y)
    return logreg


def get_model_and_visits():
    # import pre-processed data
    df_t_stay = import_data_preprocessed()

    # fit the conversion rate model
    logreg = logistic_conversion_rate(df_t_stay)

    # create visits_curve
    df_visits = create_visits_curves(
        df_t_stay[df_t_stay.T_stay.dt.date == ARRIVAL_DATE]
    )
    return logreg, df_visits


def get_reservations_from_model(ADR, logreg, df_visits):
    advance_vect = np.arange(1, 366)

    grid = np.zeros((365, 2))
    grid[:, 0] = ADR  # ADR
    grid[:, 1] = advance_vect  # Advance

    CR = logreg.predict_proba(grid)[:, 1]
    CR = np.reshape(CR, (1, 365))

    reservations = df_visits * CR
    reservations = np.flip(np.cumsum(np.flip(reservations), axis=1))
    reservations = np.clip(reservations, 0, 32)
    return advance_vect, reservations


if __name__ == "__main__":
    # df_t_stay = import_data()
    # # export to csv
    # df_t_stay.to_csv("./data/requests_t_stay.csv", index=False)

    logreg, df_visits = get_model_and_visits()

    advance_vect, reservations = get_reservations_from_model(
        ADR=100, logreg=logreg, df_visits=df_visits.values
    )

    import plotly.graph_objects as go

    # # plot the reservations curve
    # fig = go.Figure()
    # fig.add_trace(
    #     go.Scatter(x=advance_vect[75:], y=reservations[0, 75:], line=dict(width=4))
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=advance_vect[:75],
    #         y=reservations[0, :75],
    #         line=dict(color="firebrick", width=4, dash="dash"),
    #     )
    # )
    # fig.update_layout(
    #     title="Reservations curve",
    #     xaxis_title="Advance",
    #     yaxis_title="Reservations",
    #     font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
    # )
    # fig.update_yaxes(side="right", range=[0, 32])
    # fig.update_xaxes(range=[200, 0])
    # # take out legend
    # fig.update_layout(showlegend=False)
    fig_visits = go.Figure()
    fig_visits.add_trace(
        go.Scatter(
            x=np.arange(1, 366),
            y=np.flip(np.cumsum(np.flip(df_visits.values))),
            line=dict(color="black", width=4),
        )
    )
    fig_visits.update_layout(
        title="Visits curve",
        xaxis_title="Lead time",
        yaxis_title="Cumulative visits",
    )
    fig_visits.update_yaxes(side="right")
    fig_visits.update_xaxes(range=[200, 0])
    # take out legend
    fig_visits.update_layout(showlegend=False)
    fig_visits.show()
