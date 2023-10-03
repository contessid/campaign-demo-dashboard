import numpy as np
import plotly.graph_objects as go

# Create figure
fig = go.Figure()

# Range of sliders
ADR_range = np.arange(50, 350, 1)
CPC_range = np.arange(0.01, 1, 0.01)
# Other ranges
advance = np.arange(0, 100, 1)

# Add traces, one for each slider step
for ADR_step in ADR_range:
    fig.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color="#00CED1", width=6),
            # name="𝜈 = " + str(step),
            x=advance,
            y=1 / (1 + np.exp(-advance / ADR_step * 50 + 10)),
        )
    )

# Make 10th trace visible
fig.data[10].visible = True

# Create and add slider
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[
            {"visible": [False] * len(fig.data)},
            {"title": "ADR: " + str(ADR_range[i])},
        ],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [
    dict(active=10, currentvalue={"prefix": "ADR: "}, pad={"t": 50}, steps=steps),
    # dict(active=10, currentvalue={"prefix": "CPC: "}, pad={"t": 130}, steps=steps),
]

fig.update_layout(sliders=sliders)

fig.show()
