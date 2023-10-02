import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt  # Import Matplotlib

# Load the mpg dataset from Seaborn
mpg_data = sns.load_dataset('mpg')

# Create a Streamlit app
st.write("""
# MPG (Miles Per Gallon) Dataset

This dataset contains information about the fuel efficiency of various cars.

Let's visualize the relationship between horsepower and miles per gallon with a scatter plot!
""")

# Create a scatter plot using Matplotlib
plt.figure(figsize=(8, 6))  # Set the figure size

# Scatter plot
plt.scatter(mpg_data['horsepower'], mpg_data['mpg'], c='blue', alpha=0.5)
plt.title('Scatter Plot of Horsepower vs. MPG')
plt.xlabel('Horsepower')
plt.ylabel('MPG')

# Display the Matplotlib plot in Streamlit
st.pyplot(plt)
