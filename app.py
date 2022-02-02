import streamlit as st

import numpy as np
import pandas as pd

from volume import TotalMechanicalVolumeOfHUD, MirrorFullHeight

from surplus_value import SurplusValue_New

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

class scenario():
    def __init__(self, name, x1, x2, x3):
        self.name = name
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

class technology():
    def __init__(self, name, x1, x2, x3):
        self.name = name
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

class platform():
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
        s1_x2 = col3.slider("S1: Unit cost (k€/mm)", min_value=1, value=1, max_value=5)
        s1_x3 = col4.slider("S1: Redesign cost (k€/l)", min_value=1000, value=1000, max_value=5000)
    with s2_container:
        col1, col2, col3, col4 = st.columns((1, 1, 1, 1))
        col1.subheader("Scenario 2")
        s2_x1 = col2.slider("S2: Expected performance", min_value=10, value=20, max_value=30)
        s2_x2 = col3.slider("S2: Unit cost (k€/mm)", min_value=1, value=2, max_value=5)
        s2_x3 = col4.slider("S2: Redesign cost (k€/l)", min_value=1000, value=3000, max_value=5000)
    with s3_container:
        col1, col2, col3, col4 = st.columns((1, 1, 1, 1))
        col1.subheader("Scenario 3")
        s3_x1 = col2.slider("S3: Expected performance", min_value=10, value=30, max_value=30)
        s3_x2 = col3.slider("S3: Unit cost (k€/mm)", min_value=1, value=5, max_value=5)
        s3_x3 = col4.slider("S3: Redesign cost (k€/l)", min_value=1000, value=5000, max_value=5000)

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


# Display results

s1 = scenario("Scenario 1", s1_x1, s1_x2, s1_x3)
s2 = scenario("Scenario 2", s2_x1, s2_x2, s2_x3)
s3 = scenario("Scenario 3", s3_x1, s3_x2, s3_x3)

t1 = technology("Technology 1", t1_x1, t1_x2, t1_x3)
t2 = technology("Technology 2", t2_x1, t2_x2, t2_x3)
t3 = technology("Technology 3", t3_x1, t3_x2, t3_x3)

p1 = platform("Platform 1", p1_x1, p1_x2, p1_x3)
p2 = platform("Platform 2", p2_x1, p2_x2, p2_x3)
p3 = platform("Platform 3", p3_x1, p3_x2, p3_x3)

scenarios = [s1, s2, s3]
technologies = [t1, t2, t3]
platforms = [p1, p2, p3]

# initialize list of lists
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
            surplusValue = SurplusValue_New(t.x1, t.x2, mirrorSize, volume,s.x1, s.x2, s.x3)
            #st.write(f"{s.name} {t.name} {p.name} / Volume = {volume:.2f} liters / Mirror = {mirrorSize:.2f} mm / Surplus Value = {surplusValue:.2f} k€")
            row.append(surplusValue)
    data.append(row)

# Create the pandas DataFrame
#df_combinations = pd.DataFrame(data, columns = ['Scenario', 'Technology', 'Platform', 'SurplusValue'])
df_combinations = pd.DataFrame(data, columns = ['Scenario', 'T1P1', 'T1P2', 'T1P3', 'T2P1', 'T2P2', 'T2P3', 'T3P1', 'T3P2', 'T3P3'])

st.dataframe(df_combinations.style.highlight_max(axis=1))

# Tests

for p in platforms:
    st.write(f"Reserved volume: {(p.x1*p.x2*p.x3)/1_000_000} liters")


# Footer

footer_container = st.container()

with footer_container:
    footer_expander = st.expander("Debugging details")

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

