
# #######################
# ##                   ##
# ##  ICA: RBF-NN App  ##
# ##  CMSE 830         ##
# ##  12 & 14 Oct 2022 ##
# ##                   ##
# #######################


# # import libraries
# import streamlit as st
# import seaborn as sns
# import pandas as pd
# import altair as alt
# import numpy as np

# # get data
# df_mpg = sns.load_dataset("mpg")
# # st.write(df_mpg)

# labels_mpg = df_mpg.select_dtypes(include='float64').columns  # feel free to change this
# # st.write(labels_mpg)


# #######################
# ## streamlit sidebar ##
# #######################

# st.sidebar.title("""
#                  Regression
#                  Can you find the best parameters using the sliders?
#                  """)

# # allow user to choose which portion of the data to explore
# x_axis_choice = st.sidebar.selectbox(
#     "x axis",
#     labels_mpg)
# y_axis_choice = st.sidebar.selectbox(
#     "y axis",
#     labels_mpg)

# # as the user explores different parts of the data, the
# # range of the plots will change; think about how to scale
# # the plots and the ranges of the sliders to account for 
# # what the user has chosen to explore
# min_x = df_mpg.min(axis=0)[x_axis_choice]
# max_x = df_mpg.max(axis=0)[x_axis_choice]
# min_y = df_mpg.min(axis=0)[y_axis_choice]
# max_y = df_mpg.max(axis=0)[y_axis_choice]


# ##########
# ## LINE ##
# ##########

# # we will compare two models: a line and the RBF-NN; this code
# # does the line, which is done for you (but, update and change
# # as you see fit!)

# # adding the min and max side bar to enable the use enter any min and max value
# st.sidebar.write('Select the Min and Max for the line and intercept: ')
# # we want the minimum value to be consistent with the min and max of the dataset
# if min_x < min_y: 
#     minimum = min_x
# else:
#     minimum = min_y

# if max_x > max_y: 
#     maximum = max_x
# else:
#     maximum = max_y


# min_val = st.sidebar.selectbox('Min: ', np.arange(minimum, maximum))
# max_val = st.sidebar.selectbox('Max: ', np.arange(minimum, maximum))

# st.sidebar.write('line: slope and intercept')

# parameter_list=['slope','intercept']
# parameter_input_values=[]
# parameter_default_values=['0.0', '0.0']
# values=[]

# # line sliders
# # loop over the sliders: this is handy when there are many sliders
# # improvement: make the min and max values correspond to what the user selected
# for parameter, parameter_df in zip(parameter_list, parameter_default_values):
 
#     values = st.sidebar.slider(label=parameter, key=parameter, value=float(parameter_df), min_value=min_val, max_value=max_val, step=0.1)
#     parameter_input_values.append(values)
 
# input_variables = pd.DataFrame([parameter_input_values], columns=parameter_list, dtype=float)

# # calculate line
# # this uses the slider values to compute our model (a line in this case)
# # hint: copy this below and replace the line with the sum of two Gaussians
# x = np.linspace(min_x, max_x)
# line_df = pd.DataFrame({
#     'x': x,
#     'y': input_variables['slope'].iloc[0]*x + input_variables['intercept'].iloc[0]})
#     #'y': input_variables['width1']


# ############
# ## RBF-NN ##
# ############
# # adding the flexibility of changing the range of values based on the min and max of the values in the dataset
# st.sidebar.write('Select the Min and Max for the guassian parameters: ')

# min_valG = st.sidebar.selectbox('Minimum: ', np.arange(minimum, maximum))
# max_valG = st.sidebar.selectbox('Maximum: ', np.arange(minimum, maximum))

# # this is the part you need to write, starting with this starter code
# st.sidebar.write('RBF: centers, widths and heights')

# rbf_parameter_list = ['height 1', 'center 1','width 1', 'height 2', 'center 2','width 2']
# rbf_parameter_input_values = []
# rbf_parameter_default_values = ['0.0', '0.0','0.0', '0.0','0.0', '0.0']
# rbf_values = []

# # RBF-NN sliders
# # the RBF-NN model has six parameters, so you need six sliders; again, feel
# # free to upgrade this as you see fit


# for rbf_parameter, rbf_parameter_df in zip(rbf_parameter_list, rbf_parameter_default_values):
 
#     rbf_values = st.sidebar.slider(label=rbf_parameter, key=rbf_parameter, value=float(rbf_parameter_df), min_value=min_valG, max_value=max_valG, step=0.1)
#     rbf_parameter_input_values.append(rbf_values)
 
# rbf_input_variables = pd.DataFrame([rbf_parameter_input_values], columns=rbf_parameter_list, dtype=float)

# # calculate RBF-NN
# # this is just a copy of the code above, so it is just another line; replace this
# # with a sum over two Gaussians

# # For easy computing I'd be assigning variables
# h1 = rbf_input_variables['height 1'].iloc[0]
# c1 = rbf_input_variables['center 1'].iloc[0]
# l1 = rbf_input_variables['width 1'].iloc[0]

# # second guassian equation
# h2 = rbf_input_variables['height 2'].iloc[0]
# c2 = rbf_input_variables['center 2'].iloc[0]
# l2 = rbf_input_variables['width 2'].iloc[0]


# rbf_df = pd.DataFrame({
#     'x': x,
#     #'y': rbf_input_variables['center 1'].iloc[0]*x +( rbf_input_variables['center 2'].iloc[0]})
#     'y': h1*np.exp(-(x - c1)**2 / l1**2) + h2*np.exp(-(x- c2)**2 / l2**2)})


# ##############
# ## plotting ##
# ##############

# # this part is self explanatory; feel free to upgrade this simple
# # visualization to something more interesting and useful to the user
# linear_reg = alt.Chart(line_df).mark_line().encode(
#     x='x',
#     y='y',
#     color=alt.value("#FFAA00"))

# rbf_reg = alt.Chart(rbf_df).mark_line().encode(
#     x='x',
#     y='y',
#     color=alt.value("#00FF00"))

# scatter = alt.Chart(df_mpg).mark_circle(size=100).encode(x=x_axis_choice, y=y_axis_choice, color='mpg',
#     tooltip=['cylinders', 'origin', 'name']).interactive()

# scatter + linear_reg + rbf_reg

