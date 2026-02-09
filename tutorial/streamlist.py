# streamlit_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

a = st.slider("Amplitude", 0.1, 5.0, 1.0)
freq = st.slider("Frequency", 0.1, 5.0, 1.0)

x = np.linspace(0, 10, 400)
y = a * np.sin(freq * x)

fig, ax = plt.subplots()
ax.plot(x, y)
st.pyplot(fig)
