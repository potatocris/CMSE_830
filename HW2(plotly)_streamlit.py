import streamlit as st
import seaborn as sns
import pandas as pd
import plotly.express as px  # Import Plotly
from sklearn.datasets import load_iris  # Use Iris dataset

# Load the Iris dataset from Seaborn
iris_data = sns.load_dataset('iris')

# Create a Streamlit app
st.write("""
# Iris Flower Dataset

This dataset contains information about iris flowers.
It includes sepal length, sepal width, petal length, and petal width.

Let's visualize the Iris dataset with a 3D scatter plot!
""")

# Create a 3D scatter plot using Plotly
fig = px.scatter_3d(iris_data, x='sepal_length', y='sepal_width', z='petal_length', color='species',
                     symbol='species', opacity=0.6, title="3D Scatter Plot of Iris Dataset")

# Customize the layout
fig.update_layout(scene=dict(xaxis_title='Sepal Length', yaxis_title='Sepal Width', zaxis_title='Petal Length'))

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)