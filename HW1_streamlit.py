import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_wine
import altair as alt
wine_data = load_wine()
labels = wine_data.feature_names
targets = wine_data.target
print(labels)
df_form = pd.DataFrame(wine_data.data, columns = labels)
df_form['targets'] = targets
st.write("""
# Italian Wine Dataset
How are malic acid and alcohol correlated in Italian wines?
""")
alt_handle = alt.Chart(df_form).mark_circle(size=60).encode(x='alcohol', y='malic_acid',
	color='hue', tooltip=['ash', 'magnesium',
	'proanthocyanins']).interactive()
st.altair_chart(alt_handle)