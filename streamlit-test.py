import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

fig = plt.figure(figsize = [10,10])
ax = fig.add_subplot(1,1,1)

Sankey(ax=ax,  flows = [ 20400,3000,-19900,-400,-2300,-800],
              labels = ['K',   'S', 'H',   'F', 'Sp',  'x'],
        orientations = [ 1,    -1,   1,     0,   -1,   -1 ],
        scale=1/25000, trunklength=1,
        edgecolor = '#099368', facecolor = '#099368'
      ).finish()
plt.axis("off")
#plt.show()
st.plotly_chart(fig)


# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")