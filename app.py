import streamlit as st

import numpy as np
import pandas as pd

from volume import TotalMechanicalVolumeOfHUD, MirrorFullHeight

from surplus_value import surplus_value_new

# Streamlit configuration

st.set_page_config(
    page_title="VISP Method Demo",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state ="collapsed",
)

# State management

if "designs" not in st.session_state:
    st.session_state.designs = []

# Classes

class Scenario():
    def __init__(self, name, x1, x2, x3):
        self.name = name
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

class Technology():
    def __init__(self, name, x1, x2, x3):
        self.name = name
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

class Platform():
    def __init__(self, name, x1, x2, x3):
        self.name = name
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3


# ---- SIDEBAR ----

with st.sidebar:
    st.title("VISP Method Demo")

    st.write("An interactive demonstration of the VISP method")

    st.subheader("About")

    st.markdown("Learn more: [*SED Group on Github*](https://github.com/sed-group)")

# ---- MAINPAGE ----

# Configure scenarios

scenarios_expander = st.expander("Configure scenarios")

with scenarios_expander:
    s1_container = st.container()
    s2_container = st.container()
    s3_container = st.container()
    with s1_container:
        col1, col2, col3, col4 = st.columns((1, 1, 1, 1))
        col1.subheader("Scenario 1")
        s1_x1 = col2.slider("S1: Expected performance", min_value=10, value=10, max_value=30)
        s1_x2 = col3.slider("S1: Unit cost (k€/mm)", min_value=1, value=1, max_value=5) / 1000
        s1_x3 = col4.slider("S1: Redesign cost (k€/l)", min_value=0, value=1, max_value=5)
    with s2_container:
        col1, col2, col3, col4 = st.columns((1, 1, 1, 1))
        col1.subheader("Scenario 2")
        s2_x1 = col2.slider("S2: Expected performance", min_value=10, value=20, max_value=30)
        s2_x2 = col3.slider("S2: Unit cost (k€/mm)", min_value=1, value=2, max_value=5) / 1000
        s2_x3 = col4.slider("S2: Redesign cost (k€/l)", min_value=0, value=3, max_value=5)
    with s3_container:
        col1, col2, col3, col4 = st.columns((1, 1, 1, 1))
        col1.subheader("Scenario 3")
        s3_x1 = col2.slider("S3: Expected performance", min_value=10, value=30, max_value=30)
        s3_x2 = col3.slider("S3: Unit cost (k€/mm)", min_value=1, value=5, max_value=5) / 1000
        s3_x3 = col4.slider("S3: Redesign cost (k€/l)", min_value=0, value=5, max_value=5)

# Configure technologies

technologies_expander = st.expander("Configure technologies")

with technologies_expander:
    t1_container = st.container()
    t2_container = st.container()
    t3_container = st.container()
    with t1_container:
        col1, col2, col3, col4 = st.columns((1, 1, 1, 1))
        col1.subheader("Technology 1")
        t1_x1 = col2.slider("T1: Horizontal FoV", min_value=5, value=5, max_value=15)
        t1_x2 = col3.slider("T1: Vertical FoV", min_value=2, value=2, max_value=6)
        t1_x3 = col4.slider("T1: Image Distance", min_value=10000, value=10000, max_value=30000)
    with t2_container:
        col1, col2, col3, col4 = st.columns((1, 1, 1, 1))
        col1.subheader("Technology 2")
        t2_x1 = col2.slider("T2: Horizontal FoV", min_value=5, value=10, max_value=15)
        t2_x2 = col3.slider("T2: Vertical FoV", min_value=2, value=4, max_value=6)
        t2_x3 = col4.slider("T2: Image Distance", min_value=10000, value=20000, max_value=30000)
    with t3_container:
        col1, col2, col3, col4 = st.columns((1, 1, 1, 1))
        col1.subheader("Technology 3")
        t3_x1 = col2.slider("T3: Horizontal FoV", min_value=5, value=15, max_value=15)
        t3_x2 = col3.slider("T3: Vertical FoV", min_value=2, value=6, max_value=6)
        t3_x3 = col4.slider("T3: Image Distance", min_value=10000, value=30000, max_value=30000)

# Configure platforms

platforms_expander = st.expander("Configure platforms")

with platforms_expander:
    p1_container = st.container()
    p2_container = st.container()
    p3_container = st.container()
    with p1_container:
        col1, col2, col3, col4 = st.columns((1, 1, 1, 1))
        col1.subheader("Platform 1")
        p1_x1 = col2.slider("P1: Space reservation in X (mm)", min_value=100, value=300, max_value=400)
        p1_x2 = col3.slider("P1: Space reservation in Y (mm)", min_value=100, value=200, max_value=400)
        p1_x3 = col4.slider("P1: Space reservation in Z (mm)", min_value=100, value=150, max_value=400)
    with p2_container:
        col1, col2, col3, col4 = st.columns((1, 1, 1, 1))
        col1.subheader("Platform 2")
        p2_x1 = col2.slider("P2: Space reservation in X (mm)", min_value=100, value=400, max_value=400)
        p2_x2 = col3.slider("P2: Space reservation in Y (mm)", min_value=100, value=200, max_value=400)
        p2_x3 = col4.slider("P2: Space reservation in Z (mm)", min_value=100, value=200, max_value=400)
    with p3_container:
        col1, col2, col3, col4 = st.columns((1, 1, 1, 1))
        col1.subheader("Platform 3")
        p3_x1 = col2.slider("P3: Space reservation in X (mm)", min_value=100, value=300, max_value=400)
        p3_x2 = col3.slider("P3: Space reservation in Y (mm)", min_value=100, value=200, max_value=400)
        p3_x3 = col4.slider("P3: Space reservation in Z (mm)", min_value=100, value=300, max_value=400)

s1 = Scenario("Scenario 1", s1_x1, s1_x2, s1_x3)
s2 = Scenario("Scenario 2", s2_x1, s2_x2, s2_x3)
s3 = Scenario("Scenario 3", s3_x1, s3_x2, s3_x3)

t1 = Technology("Technology 1", t1_x1, t1_x2, t1_x3)
t2 = Technology("Technology 2", t2_x1, t2_x2, t2_x3)
t3 = Technology("Technology 3", t3_x1, t3_x2, t3_x3)

p1 = Platform("Platform 1", p1_x1, p1_x2, p1_x3)
p2 = Platform("Platform 2", p2_x1, p2_x2, p2_x3)
p3 = Platform("Platform 3", p3_x1, p3_x2, p3_x3)

scenarios = [s1, s2, s3]
technologies = [t1, t2, t3]
platforms = [p1, p2, p3]

# Calculate the surplus value for each combination of scenarios, technologies and platforms
data = []
for i, s in enumerate(scenarios):
    row = [f'Scenario {i+1}']
    for t in technologies:
        for p in platforms:
            inputs = {
                "FullHorizontalFOV" : t.x1,
                "FullVerticalFOV" : t.x2,
                "VirtualImageDistance" : t.x3,
                "EyeboxToMirror1" : 1000,
                "EyeboxFullWidth" : 140,
                "EyeboxFullHeight" : 60,
                "Mirror1ObliquityAngle" : 30,
                "HUD_SCREEN_10x5_FOV_BASELINE_WIDTH" : 70,
                "MechanicalVolumeIncrease" : 40,
                "M1M2OverlapFraction" : 0,
                "PGUVolumeEstimate" : 0.5
            }
            volume = TotalMechanicalVolumeOfHUD(inputs)
            mirrorSize = MirrorFullHeight(inputs)
            reservedVolume = (p.x1*p.x2*p.x3)/1_000_000
            volumeUsePercentage = volume / reservedVolume * 100
            # st.write(f"Use of volume = {volumeUsePercentage:.2f} % ____ {volume:.2f} of {reservedVolume:.2f}")
            surplusValue = surplus_value_new(t.x1, t.x2, mirrorSize, volume, s.x1, s.x2, s.x3)
            #st.write(f"{s.name} {t.name} {p.name} / Volume = {volume:.2f} liters / Mirror = {mirrorSize:.2f} mm / Surplus Value = {surplusValue:.2f} k€")
            row.append(surplusValue)
    data.append(row)

data_tech = [['Scenario 1',(data[0][1]+data[0][2]+data[0][3])/3,(data[0][4]+data[0][5]+data[0][6])/3,(data[0][7]+data[0][8]+data[0][9])/3],
             ['Scenario 2',(data[1][1]+data[1][2]+data[1][3])/3,(data[1][4]+data[1][5]+data[1][6])/3,(data[1][7]+data[1][8]+data[1][9])/3],
             ['Scenario 3',(data[2][1]+data[2][2]+data[2][3])/3,(data[2][4]+data[2][5]+data[2][6])/3,(data[2][7]+data[2][8]+data[2][9])/3]]

data_plat_avg = [['Scenario 1',(data[0][2]+data[0][4]+data[0][7])/3,(data[0][2]+data[0][5]+data[0][8])/3,(data[0][3]+data[0][6]+data[0][9])/3],
                 ['Scenario 2',(data[1][2]+data[1][4]+data[1][7])/3,(data[1][2]+data[1][5]+data[1][8])/3,(data[1][3]+data[1][6]+data[1][9])/3],
                 ['Scenario 3',(data[2][2]+data[2][4]+data[2][7])/3,(data[2][2]+data[2][5]+data[2][8])/3,(data[2][3]+data[2][6]+data[2][9])/3]]

data_plat_max = [['Scenario 1',max([data[0][2],data[0][4],data[0][7]]),max([data[0][2],data[0][5],data[0][8]]),max([data[0][3],data[0][6],data[0][9]])],
                 ['Scenario 2',max([data[1][2],data[1][4],data[1][7]]),max([data[1][2],data[1][5],data[1][8]]),max([data[1][3],data[1][6],data[1][9]])],
                 ['Scenario 3',max([data[2][2],data[2][4],data[2][7]]),max([data[2][2],data[2][5],data[2][8]]),max([data[2][3],data[2][6],data[2][9]])]]

# Create the pandas DataFrames
#df_combinations = pd.DataFrame(data, columns = ['Scenario', 'Technology', 'Platform', 'SurplusValue'])
df_combinations = pd.DataFrame(data, columns = ['Scenario', 'T1P1', 'T1P2', 'T1P3', 'T2P1', 'T2P2', 'T2P3', 'T3P1', 'T3P2', 'T3P3'])
df_tech = pd.DataFrame(data_tech, columns = ['Scenario', 'T1', 'T2', 'T3'])
df_plat_avg = pd.DataFrame(data_plat_avg, columns = ['Scenario', 'P1', 'P2', 'P3'])
df_plat_max = pd.DataFrame(data_plat_max, columns = ['Scenario', 'P1', 'P2', 'P3'])

# Display highlights

high_col1, high_col2, high_col3, high_col4 = st.columns((1, 1, 1, 1))

high_col1.metric("Highest value", "300", delta=None, delta_color="normal")
high_col2.metric("Scenario", "1", delta=None, delta_color="normal")
high_col3.metric("Technology", "2", delta=None, delta_color="normal")
high_col4.metric("Platform", "3", delta=None, delta_color="normal")

# Display the data tables
st.write("The highlighted cells are the maximum Surplus Value for each scenario, for TxPy (Technology x Platform y):")
st.dataframe(df_combinations.style.highlight_max(axis=1))

st.write("The highlighted cells are the maximum Surplus Value for each scenario, for Tx (Technology x):")
st.dataframe(df_tech.style.highlight_max(axis=1))

st.write("AVERAGES - The highlighted cells are the maximum Surplus Value for each scenario, for Py (Platform y):")
st.dataframe(df_plat_avg.style.highlight_max(axis=1))

st.write("MAXIMUM - The highlighted cells are the maximum Surplus Value for each scenario, for Py (Platform y):")
st.dataframe(df_plat_max.style.highlight_max(axis=None))

# Tests

# for p in platforms:
#     st.write(f"Reserved volume: {(p.x1*p.x2*p.x3)/1_000_000} liters")


# Footer

# footer_container = st.container()

# with footer_container:
#     footer_expander = st.expander("Debugging details")

# with footer_expander:
#     st.write(st.session_state.designs)


# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                header {visibility: visible;}
                footer {visibility: hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)

