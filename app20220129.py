import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import plotly.express as px

from operator import lt as less_than, gt as greater_than
from operator import (truediv as div, mul)

from volume import TotalMechanicalVolumeOfHUD, MirrorFullHeight

from surplus_value import SurplusValue

st.set_page_config(
    page_title="VISP Method Demo",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "designs" not in st.session_state:
    st.session_state.designs = []

def add_design(design):
    st.session_state.designs.append(design)

"""
# VISP Method Demo

**An interactive demonstration of the VISP method**

TODO Description here

"""

#expander_configuration = st.beta_expander("Configuration")

#with expander_configuration: config_col1, config_col2, config_col3, config_col4 = st.beta_columns(4)
config_col1, config_col2, config_col3, config_col4 = st.beta_columns((1, 1, 1, 2))

with config_col1: FullHorizontalFOV = st.slider('FullHorizontalFOV', min_value=5, value=10, max_value=15)
with config_col1: FullVerticalFOV = st.slider('FullVerticalFOV', min_value=2, value=3, max_value=6)
with config_col1: EyeboxFullWidth = st.slider('EyeboxFullWidth', min_value=70, value=100, max_value=210)
with config_col1: EyeboxFullHeight = st.slider('EyeboxFullHeight', min_value=30, value=50, max_value=90)
with config_col2: VirtualImageDistance = st.slider('VirtualImageDistance', min_value=10000, value=10000, max_value=30000)
with config_col2: EyeboxToMirror1 = st.slider('EyeboxToMirror1', min_value=500, value=600, max_value=1500)
with config_col2: Mirror1ObliquityAngle = st.slider('Mirror1ObliquityAngle', min_value=15, value=20, max_value=45)

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
        "PGUVolumeEstimate": 0.5
    }

numberDesigns = st.sidebar.number_input('Set number of designs', 10, 1000, 100)

totalVolume = TotalMechanicalVolumeOfHUD(inputs)
mirrorSize = MirrorFullHeight(inputs)
surplusValue = SurplusValue(FullHorizontalFOV, FullVerticalFOV, mirrorSize, totalVolume)

design = {"inputs": inputs, "mirrorSize": mirrorSize, "totalVolume": totalVolume, "surplusValue": surplusValue}
add_design(design)

with config_col3: st.write(f'Total volume = {totalVolume}')
with config_col3: st.write(f'Mirror size = {mirrorSize}')
with config_col3: st.write(f'Surplus value = {surplusValue}')

st.write(st.session_state.designs)

def add_designs():
    i=0
    while i < numberDesigns:
        inputs = {
                "FullHorizontalFOV": random.uniform(5, 15),
                "FullVerticalFOV": random.uniform(2, 6),
                "EyeboxFullWidth": random.uniform(70, 210),
                "EyeboxFullHeight": random.uniform(30, 90),
                "VirtualImageDistance": random.uniform(10000, 30000),
                "EyeboxToMirror1": random.uniform(500, 1500),
                "Mirror1ObliquityAngle": random.uniform(15, 45),
                "HUD_SCREEN_10x5_FOV_BASELINE_WIDTH": 70,
                "MechanicalVolumeIncrease": 20,
                "M1M2OverlapFraction": 0,
                "PGUVolumeEstimate": 0.5
            }

        totalVolume = TotalMechanicalVolumeOfHUD(inputs)
        if totalVolume <= 0:
            print(inputs)
        mirrorSize = MirrorFullHeight(inputs)
        surplusValue = SurplusValue(inputs["FullHorizontalFOV"], inputs["FullVerticalFOV"], mirrorSize, totalVolume)
        design = {"inputs": inputs, "mirrorSize": mirrorSize, "totalVolume": totalVolume, "surplusValue": surplusValue}
        if totalVolume > 0:
            add_design(design)
            i = i + 1

st.sidebar.button(f'Add {numberDesigns} designs', on_click=add_designs)

feature1, feature2 = "Volume (liter)", "Surplus Value (k€)"

#mode_ = st.sidebar.selectbox(f'Optimisation direction of {feature1}, {feature2}', ["min, max","min, min","max, min", "max, max"])
mode_ = "min, max"

def generate_objectives():

    volumes = []
    values = []
    for design in st.session_state.designs:
        volumes.append(design["totalVolume"])
        values.append(design["surplusValue"])

    return {feature1: volumes, feature2: values}

#objective_values = generate_objectives(n_packages, seed=seed, value_distribution_mode=value_distribution_mode, visualise=False)
objective_values = generate_objectives()

if mode_ == "min, max":
    objective_mode, heuristic, soh_unit = {feature1: 'min', feature2: 'max'}, 'value/volume', 'k€/liter'
elif mode_ == "min, min":
    objective_mode, heuristic, soh_unit = {feature1: 'min', feature2: 'min'}, '1/value/volume', '1/k€/liter'
elif mode_ == "max, max":
    objective_mode, heuristic, soh_unit = {feature1: 'max', feature2: 'max'}, 'value*volume',  'k€*liter'
elif mode_ == "max, min":
    objective_mode, heuristic, soh_unit = {feature1: 'max', feature2: 'min'}, 'volume/value', 'liter/k€'


# for Pareto Optimal selection
mode_to_operator = {'min': less_than, 'max': greater_than}
objective_operator = {key: mode_to_operator[objective_mode[key]] for key in objective_mode.keys()}

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

        other_weights = np.array(list(objective_values[feature1][:idx]) + list(
            objective_values[feature1][idx + 1:]))
        other_values = np.array(list(objective_values[feature2][:idx]) + list(
            objective_values[feature2][idx + 1:]))

        for jdx in range(len(other_weights)):
            other_dominates = objective_operator[feature1](other_weights[jdx],
                                                           this_weight) & \
                              objective_operator[feature2](other_values[jdx],
                                                           this_value)

            if other_dominates:
                is_pareto = False
                break

        if is_pareto:
            idxs_pareto.append(idx_objects[idx])

    return idxs_pareto


pareto_idxs = objectives_to_pareto_front(objective_values)
print(f'Pareto front: {pareto_idxs}')
for pareto_idx in pareto_idxs:
    print(f'Pareto point: {objective_values[feature1][pareto_idx]}, {objective_values[feature2][pareto_idx]}')
    print(next((item for item in st.session_state.designs if item["totalVolume"] == objective_values[feature1][pareto_idx]), None))

# plt.scatter(objective_values[feature1], objective_values[feature2], s=10, alpha=0.7, color='blue')

# plt.scatter([val for idx, val in enumerate(objective_values[feature1]) if idx in pareto_idxs],
#             [val for idx, val in enumerate(objective_values[feature2]) if idx in pareto_idxs],
#             marker='x', s=100, linewidth=4, color='green')

list1 = [val for idx, val in enumerate(objective_values[feature1]) if idx in pareto_idxs]
list2 = [val for idx, val in enumerate(objective_values[feature2]) if idx in pareto_idxs]
list1, list2 = zip(*sorted(zip(list1, list2)))
# To return lists instead of tuples, uncomment the following line:
# list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))

# plt.plot(list1,
#         list2,
#         marker='x', linewidth=4, color='red')

# plt.xlabel(feature1)
# plt.ylabel(feature2)


# with config_col4: st.pyplot(plt.gcf())

"""
Learn more: [*SED Group on Github*](https://github.com/sed-group)  
"""

## Plotly

designs_df = pd.DataFrame(st.session_state.designs)

plot = fig = px.scatter(
    data_frame=designs_df,
    x="totalVolume",
    y="surplusValue",
    #color="surplusValue",
    labels="Alternative designs",
)

fig.add_scatter(x=list1, y=list2, name="Pareto front")

with config_col4: st.plotly_chart(plot, use_container_width=True)

