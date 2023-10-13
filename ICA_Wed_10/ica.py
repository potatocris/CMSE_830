import streamlit as st
import seaborn as sns
import pandas as pd
import altair as alt
import numpy as np

ds = np.array([[0.5, 2], [1, 1]])

# Select the dataset to use
dataset_choice = st.sidebar.radio(
    "**Select Dataset**", ("mpg", "iris", "Provided Data"))

# Load Dataset
if dataset_choice == "mpg":
    df = sns.load_dataset("mpg")
elif dataset_choice == "iris":
    df = sns.load_dataset("iris")
else:
    df = pd.DataFrame(ds, columns=["x", "y"])

# Load the dataset and select numerical columns
numeric_col = df.select_dtypes(include='float64').columns

# Sidebar controls

# Select x and y axes for exploration
x_axis_choice = st.sidebar.selectbox("**X-Axis**", numeric_col)
y_axis_choice = st.sidebar.selectbox("**Y-Axis**", numeric_col)

st.sidebar.write('You selected:', x_axis_choice, "and", y_axis_choice)

# Define ranges for sliders based on the selected data
min_x = df.min(axis=0)[x_axis_choice]
max_x = df.max(axis=0)[x_axis_choice]
min_y = df.min(axis=0)[y_axis_choice]
max_y = df.max(axis=0)[y_axis_choice]

if min_x < min_y:
    minimum = min_x
else:
    minimum = min_y

if max_x > max_y:
    maximum = max_x
else:
    maximum = max_y


# Sidebar controls for the Line Model
st.sidebar.write('## Line Model')
st.sidebar.write('**Select the Min and Max:**')
min_val_line = st.sidebar.selectbox(
    '**Min**:', np.arange(minimum, maximum), key="lin1")
max_val_line = st.sidebar.selectbox(
    '**Max**:', np.arange(minimum, maximum), key="lin2")

st.sidebar.write('**Choose Slope and Intercept**')
slope = st.sidebar.slider('**Slope**', min_val_line, max_val_line, 0.0, 0.1)
intercept = st.sidebar.slider(
    '**Intercept**', min_val_line, max_val_line, 0.0, 0.1)

# Sidebar controls for the RBF-NN Model
st.sidebar.write('## RBF-NN Model')
st.sidebar.write('**Select the Min and Max:**')
min_val_rbf = st.sidebar.selectbox(
    '**Min**:', np.arange(minimum, maximum), key="rbf1")
max_val_rbf = st.sidebar.selectbox(
    '**Max**:', np.arange(minimum, maximum), key="rbf2")

st.sidebar.write('**Choose Gaussian Parameters**')
weigth1 = st.sidebar.slider('**Weigth 1**', min_val_rbf, max_val_rbf, 0.0, 0.1)
center1 = st.sidebar.slider('**Center 1**', min_val_rbf, max_val_rbf, 0.0, 0.1)
band1 = st.sidebar.slider(
    '**Bandwidth 1**', min_val_rbf, max_val_rbf, 0.0, 0.1)
weigth2 = st.sidebar.slider('**Weigth 2**', min_val_rbf, max_val_rbf, 0.0, 0.1)
center2 = st.sidebar.slider('**Center 2**', min_val_rbf, max_val_rbf, 0.0, 0.1)
band2 = st.sidebar.slider(
    '**Bandwidth 2**', min_val_rbf, max_val_rbf, 0.0, 0.1)

# Calculate line model
x = np.linspace(minimum, maximum)
line_df = pd.DataFrame({
    'x': x,
    'y': slope * x + intercept
})

# Calculate RBF-NN
rbf_df = pd.DataFrame({
    'x': x,
    'y': weigth1 * np.exp(-((x - center1) ** 2) / (band1 ** 2)) +
    weigth2 * np.exp(-((x - center2) ** 2) / (band2 ** 2))
})

# Main content
st.title("Regression Models")
st.write(
    f"Exploring the relationship between {x_axis_choice} and {y_axis_choice}")

# Visualization
line_chart = alt.Chart(line_df).mark_line(color='blue').encode(
    x='x',
    y='y'
)

rbf_chart = alt.Chart(rbf_df).mark_line(color='red').encode(
    x='x',
    y='y'
)

scatter = alt.Chart(df).mark_circle(size=60).encode(
    x=x_axis_choice,
    y=y_axis_choice,
    color=x_axis_choice
)

st.write((scatter + line_chart + rbf_chart).properties(width=700, height=400))

 # Calculate MAE and MSE
def MAE_MSE(slope, intercept, df):
    MAE = 0
    MSE = 0
    for i, x in enumerate(df[x_axis_choice]):
        MAE += np.abs(df[y_axis_choice][i] - (slope*x+intercept))
        MSE += (df[y_axis_choice][i] - (slope*x+intercept))**2
    return MAE, MSE

def MAE_MSE_RBF(center1,weight1,band1,center2,weight2,band2,df):
        MAE = 0
        MSE = 0
        for i,x in enumerate(df[x_axis_choice]):
            y_err=weight1 * np.exp(-((df[x_axis_choice][i] - center1) ** 2) / (band1 ** 2)) +\
            weight2 * np.exp(-((df[x_axis_choice][i] - center2) ** 2) / (band2 ** 2))
            MAE += np.abs(df[y_axis_choice][i] - y_err)
            MSE += (df[y_axis_choice][i]  - y_err)**2
        return MAE,MSE

# Display MAE and MSE based on user choice
model_choice = st.sidebar.radio("Select Model", ("Line Model", "RBF-NN Model"))

if model_choice == "Line Model":
    MAE, MSE = MAE_MSE(slope, intercept, df)
    st.write(f"MAE for Line Model: {MAE}, MSE for Line Model: {MSE}")
elif model_choice == "RBF-NN Model":
    MAE_RBF, MSE_RBF = MAE_MSE_RBF(center1, weight1, band1, center2, weight2, band2, rbf_df)
    st.write(f"MAE for RBF-NN Model: {MAE_RBF}, MSE for RBF-NN Model: {MSE_RBF}")
