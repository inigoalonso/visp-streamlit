import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

import plotly.express as px

from doepy import build

from operator import lt as less_than, gt as greater_than
from operator import truediv as div, mul

from volume import TotalMechanicalVolumeOfHUD, MirrorFullHeight

from surplus_value import SurplusValue

# Streamlit configuration

st.set_page_config(
    page_title="VISP Method Demo",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state ="auto",
)

# State management

if "designs" not in st.session_state:
    st.session_state.designs = []

# Random seed

# np.random.seed(19680801)

# Functions

def add_design(design):
    st.session_state.designs.append(design)


def generate_lhs():

    factors_ar = {
        "FullHorizontalFOV": [5, 15],
        "FullVerticalFOV": [2, 6],
        "VirtualImageDistance": [10000, 30000],
        "EyeboxToMirror1": [500, 1500],
        "EyeboxFullWidth": [70, 210],
        "EyeboxFullHeight": [30, 90],
        "Mirror1ObliquityAngle": [15, 45],
    }

    num_samples = numberDesigns

    lhs = build.space_filling_lhs(factors_ar, num_samples=num_samples)

    for index, row in lhs.iterrows():
        inputs = {
            "FullHorizontalFOV": row["FullHorizontalFOV"],
            "FullVerticalFOV": row["FullVerticalFOV"],
            "VirtualImageDistance": row["VirtualImageDistance"],
            "EyeboxToMirror1": row["EyeboxToMirror1"],
            "EyeboxFullWidth": row["EyeboxFullWidth"],
            "EyeboxFullHeight": row["EyeboxFullHeight"],
            "Mirror1ObliquityAngle": row["Mirror1ObliquityAngle"],
            "HUD_SCREEN_10x5_FOV_BASELINE_WIDTH": 70,
            "MechanicalVolumeIncrease": 20,
            "M1M2OverlapFraction": 0,
            "PGUVolumeEstimate": 0.5,
        }

        mirrorSize = MirrorFullHeight(inputs)
        totalVolume = TotalMechanicalVolumeOfHUD(inputs)
        lhs.at[index, "mirrorSize"] = mirrorSize
        lhs.at[index, "totalVolume"] = totalVolume

    #print(lhs)

    return lhs


def add_designs():
    lhs = generate_lhs()
    i = 0
    while i < numberDesigns:

        totalVolume = lhs["totalVolume"][i]
        if totalVolume <= 0:
            print(f"totalVolume = 0\nlhs=\n{lhs}")
        surplusValue = SurplusValue(
            lhs["FullHorizontalFOV"][i],
            lhs["FullVerticalFOV"][i],
            lhs["mirrorSize"][i],
            lhs["totalVolume"][i],
        )
        design = {
            "FullHorizontalFOV": lhs["FullHorizontalFOV"][i],
            "FullVerticalFOV": lhs["FullVerticalFOV"][i],
            "VirtualImageDistance": lhs["VirtualImageDistance"][i],
            "EyeboxToMirror1": lhs["EyeboxToMirror1"][i],
            "EyeboxFullWidth": lhs["EyeboxFullWidth"][i],
            "EyeboxFullHeight": lhs["EyeboxFullHeight"][i],
            "Mirror1ObliquityAngle": lhs["Mirror1ObliquityAngle"][i],
            "mirrorSize": lhs["mirrorSize"][i],
            "totalVolume": lhs["totalVolume"][i],
            "surplusValue": surplusValue,
        }
        if totalVolume > 0:
            add_design(design)
            i = i + 1


def generate_objectives():

    volumes = []
    values = []
    for design in st.session_state.designs:
        volumes.append(design["totalVolume"])
        values.append(design["surplusValue"])

    return {feature1: volumes, feature2: values}


def objectives_to_pareto_front(objective_values):
    feature1 = list(objective_values.keys())[0]
    feature2 = list(objective_values.keys())[1]

    # objective_values = {}
    # for objective in objectives:
    #    objective_values[objective] = [knapsacks[idx][objective] for idx in knapsacks]

    idxs_pareto = []

    idx_objects = np.arange(len(objective_values[feature1]))

    for idx in idx_objects:
        is_pareto = True

        this_weight = objective_values[feature1][idx]
        this_value = objective_values[feature2][idx]

        other_weights = np.array(
            list(objective_values[feature1][:idx])
            + list(objective_values[feature1][idx + 1 :])
        )
        other_values = np.array(
            list(objective_values[feature2][:idx])
            + list(objective_values[feature2][idx + 1 :])
        )

        for jdx in range(len(other_weights)):
            other_dominates = objective_operator[feature1](
                other_weights[jdx], this_weight
            ) & objective_operator[feature2](other_values[jdx], this_value)

            if other_dominates:
                is_pareto = False
                break

        if is_pareto:
            idxs_pareto.append(idx_objects[idx])

    return idxs_pareto


# ---- SIDEBAR ----

with st.sidebar:
    st.title("VISP Method Demo")

    header_expander = st.expander("Description")

    with header_expander:
        st.write("An interactive demonstration of the VISP method")

    st.header("Scenarios")
    scenario = st.multiselect(
        'Select scenario',
        ['Scenario 1', 'Scenario 2', 'Scenario 3'],
        ['Scenario 1'])

    st.header("Technologies")
    technology = st.multiselect(
        'Select technology',
        ['Technology 1', 'Technology 2', 'Technology 3'],
        ['Technology 1'])

    st.header("Platforms")
    platform = st.multiselect(
        'Select platform',
        ['Platform 1', 'Platform 2', 'Platform 3'],
        ['Platform 1'])

    st.subheader("About")

    st.markdown("Learn more: [*SED Group on Github*](https://github.com/sed-group)")

# ---- MAINPAGE ----

# Containers

plot_container = st.container()
single_design_expander = st.expander("Configure a single design")
doe_expander = st.expander("Configure a DOE batch")
footer_container = st.container()

# Single design

with single_design_expander:
    single_design_form = st.form("single_design_form")

with single_design_form:
    config_col1, config_col2, config_col3 = single_design_form.columns((1, 1, 1))

    FullHorizontalFOV = st.slider(
    "FullHorizontalFOV", min_value=5, value=10, max_value=15
    )
    FullVerticalFOV = st.slider("FullVerticalFOV", min_value=2, value=3, max_value=6)
    EyeboxFullWidth = st.slider(
    "EyeboxFullWidth", min_value=70, value=100, max_value=210
    )
    EyeboxFullHeight = st.slider(
    "EyeboxFullHeight", min_value=30, value=50, max_value=90
    )
    VirtualImageDistance = st.slider(
    "VirtualImageDistance", min_value=10000, value=10000, max_value=30000
    )
    EyeboxToMirror1 = st.slider(
    "EyeboxToMirror1", min_value=500, value=600, max_value=1500
    )
    Mirror1ObliquityAngle = st.slider(
    "Mirror1ObliquityAngle", min_value=15, value=20, max_value=45
    )

    inputs = {
        "FullHorizontalFOV": FullHorizontalFOV,
        "FullVerticalFOV": FullVerticalFOV,
        "EyeboxFullWidth": EyeboxFullWidth,
        "EyeboxFullHeight": EyeboxFullHeight,
        "VirtualImageDistance": VirtualImageDistance,
        "EyeboxToMirror1": EyeboxToMirror1,
        "Mirror1ObliquityAngle": Mirror1ObliquityAngle,
        "HUD_SCREEN_10x5_FOV_BASELINE_WIDTH": 70,
        "MechanicalVolumeIncrease": 20,
        "M1M2OverlapFraction": 0,
        "PGUVolumeEstimate": 0.5,
    }

    totalVolume = TotalMechanicalVolumeOfHUD(inputs)
    mirrorSize = MirrorFullHeight(inputs)
    surplusValue = SurplusValue(FullHorizontalFOV, FullVerticalFOV, mirrorSize, totalVolume)

    design = {
        **inputs,
        "mirrorSize": mirrorSize,
        "totalVolume": totalVolume,
        "surplusValue": surplusValue,
}

    try:
        totalVolumeDelta = st.session_state.designs[-1]['totalVolume']-st.session_state.designs[-2]['totalVolume']
        mirrorSizeDelta = st.session_state.designs[-1]['mirrorSize']-st.session_state.designs[-2]['mirrorSize']
        surplusValueDelta = st.session_state.designs[-1]['surplusValue']-st.session_state.designs[-2]['surplusValue']
    except IndexError:
        totalVolumeDelta = 0
        mirrorSizeDelta = 0
        surplusValueDelta = 0
    
    config_col1.metric("Volume", f"{totalVolume:.2f}", f"{totalVolumeDelta:.2f}", delta_color="inverse")
    config_col2.metric("Mirror size", f"{mirrorSize:.2f}", f"{mirrorSizeDelta:.2f}", delta_color="inverse")
    config_col3.metric("Surplus value", f"{surplusValue:.2f}", f"{surplusValueDelta:.2f}")

    # Every form must have a submit button.
    single_design_submitted = st.form_submit_button("Submit", on_click=add_design(design))
    if single_design_submitted:
        st.write("FullHorizontalFOV", FullHorizontalFOV)

# DOE

with doe_expander:
    doe_form = st.form("doe_form")

with doe_form:
    numberDesigns = st.number_input("Set number of designs", 10, 1000, 100)
    single_design_submitted = st.form_submit_button(f"Add {numberDesigns} designs", on_click=add_designs)


# Plot
# ---- PARETO FRONT ----
feature1, feature2 = "Volume (liter)", "Surplus Value (M€)"

# mode_ = st.sidebar.selectbox(f'Optimisation direction of {feature1}, {feature2}', ["min, max","min, min","max, min", "max, max"])
mode_ = "min, max"

# objective_values = generate_objectives(n_packages, seed=seed, value_distribution_mode=value_distribution_mode, visualise=False)
objective_values = generate_objectives()

if mode_ == "min, max":
    objective_mode, heuristic, soh_unit = (
        {feature1: "min", feature2: "max"},
        "value/volume",
        "M€/liter",
    )
elif mode_ == "min, min":
    objective_mode, heuristic, soh_unit = (
        {feature1: "min", feature2: "min"},
        "1/value/volume",
        "1/M€/liter",
    )
elif mode_ == "max, max":
    objective_mode, heuristic, soh_unit = (
        {feature1: "max", feature2: "max"},
        "value*volume",
        "M€*liter",
    )
elif mode_ == "max, min":
    objective_mode, heuristic, soh_unit = (
        {feature1: "max", feature2: "min"},
        "volume/value",
        "liter/M€",
    )


# for Pareto Optimal selection
mode_to_operator = {"min": less_than, "max": greater_than}
objective_operator = {
    key: mode_to_operator[objective_mode[key]] for key in objective_mode.keys()
}

pareto_idxs = objectives_to_pareto_front(objective_values)
# print(f"Pareto front: {pareto_idxs}")
# for pareto_idx in pareto_idxs:
#     print(
#         f"Pareto point: {objective_values[feature1][pareto_idx]}, {objective_values[feature2][pareto_idx]}"
#     )
#     print(
#         next(
#             (
#                 item
#                 for item in st.session_state.designs
#                 if item["totalVolume"] == objective_values[feature1][pareto_idx]
#             ),
#             None,
#         )
#     )

# plt.scatter(objective_values[feature1], objective_values[feature2], s=10, alpha=0.7, color='blue')

# plt.scatter([val for idx, val in enumerate(objective_values[feature1]) if idx in pareto_idxs],
#             [val for idx, val in enumerate(objective_values[feature2]) if idx in pareto_idxs],
#             marker='x', s=100, linewidth=4, color='green')

list1 = [
    val for idx, val in enumerate(objective_values[feature1]) if idx in pareto_idxs
]
list2 = [
    val for idx, val in enumerate(objective_values[feature2]) if idx in pareto_idxs
]
list1, list2 = zip(*sorted(zip(list1, list2)))


with plot_container:
    ## Plotly

    designs_df = pd.DataFrame(st.session_state.designs)

    plot = fig = px.scatter(
        data_frame=designs_df,
        x="totalVolume",
        y="surplusValue",
        #color="surplusValue",
        title="Surplus value vs Total volume of the HUD",
        labels=dict(totalVolume="Total volume (liter)", surplusValue="Surplus value (M€)")
    )

    fig.add_scatter(x=list1, y=list2, name="Pareto front")

    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))

    st.plotly_chart(plot, use_container_width=True)

    # Matplotlib

    fig, ax = plt.subplots(figsize=(6, 6))

    x = designs_df.totalVolume
    y = designs_df.surplusValue
    ax.scatter(x, y, alpha=0.3, edgecolors='none')

    ax.plot([10, 10], [designs_df.surplusValue.max(), 0], label="Line",
            path_effects=[path_effects.withTickedStroke(spacing=15, length=3)])


    ax.set_xlim(0, designs_df.totalVolume.max())
    ax.set_ylim(0, designs_df.surplusValue.max())

    #ax.legend()
    ax.grid(True)

    #st.pyplot(fig)


# Footer
footer_expander = footer_container.expander("Details of the stored designs")

with footer_expander:
    st.write(st.session_state.designs)


# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                header {visibility: visible;}
                footer {visibility: hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)

