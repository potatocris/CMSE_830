# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np
import zipcodes as zcode  # to get zipcodes

# Libraries to help with data visualization
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import streamlit as st

st.set_page_config(
    page_title="Loan Marketing",
    page_icon="🏦",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "This web application is my CMSE 830 Mid Term Project Submission. Enjoy my Hard Work!"
    },
)
st.set_option("deprecation.showPyplotGlobalUse", False)
# st.set_page_config(layout="wide")

# setting the layout for the seaborn plot
sns.set(style="darkgrid")

# I want help a bank identify customers likely to accept personal loan offers, and ultimately drive growth and profitability
file = r"Midterm_Project/draft/Loan_Modelling.csv"

# Load the dataset


@st.cache_data
def load_data(file):
    data = pd.read_csv(file, index_col="ID")

    return data


# Load the data using the defined function
data = load_data(file)
df = data.copy()

# Set Streamlit app title
st.title(":gray[Enhancing AllLife Bank's Personal Loan Marketing Strategy] 🏦")

# Add an expander
with st.expander("**Background & Context**"):
    st.markdown(
        """
    AllLife Bank aims to grow its customer base, focusing on increasing the number of borrowers (asset customers) while retaining 
    depositors (liability customers). Last year's campaign for liability customers had a conversion rate of over 9%, inspiring 
    the retail marketing department to create more efficient, targeted campaigns with a minimal budget to boost this ratio further.
    """
    )

with st.expander("**Data Dictionary**"):
    st.markdown(
        """
         * **`ID`**: Unique Customer Identification Number
         * **`Age`**: Customer’s age in years
         * **`Experience`**: Years of professional experience
         * **`Income`**: Annual income of the customer (in thousand dollars)
         * **`ZIP Code`**: Home Address ZIP code.
         * **`Family`**: Family size of the customer
         * **`CCAvg`**: Avg. spending on credit cards per month (in thousand dollars)
         * **`Education`**: Education Level. 1: Undergrad; 2: Graduate;3: Advanced/Professional
         * **`Mortgage`**: Value of house mortgage if any. (in thousand dollars)
         * **`Personal_Loan`**: Did this customer accept the personal loan offered in the last campaign?
         * **`Securities_Account`**: Does the customer have securities account with the bank?
         * **`CD_Account`**: Does the customer have a certificate of deposit (CD) account with the bank?
         * **`Online`**: Do customers use internet banking facilities?
         * **`CreditCard`**: Does the customer use a credit card issued by Universal Bank?
         """
    )

# Data Preprocessing
# --------------------------------

# The minumum value of Experience column is -3.0 so replace negative values in 'Experience' with NaN
df["Experience"] = df["Experience"].clip(lower=0)
df["Experience"] = df.groupby("Age")["Experience"].transform(
    lambda x: x.fillna(x.median())
)  # Impute missing values in 'Experience' based on median experience for each age group

# Converting Zipcode to County
list_zipcode = df.ZIPCode.unique()
dict_zip = {}
for zipcode in list_zipcode:
    city_county = zcode.matching(zipcode.astype("str"))
    if len(city_county) == 1:
        county = city_county[0].get("county")
    else:
        county = zipcode

    dict_zip.update({zipcode: county})
dict_zip.update({92717: "Orange County"})
dict_zip.update({92634: "Orange County"})
dict_zip.update({96651: "El Dorado County"})
dict_zip.update({93077: "Ventura County"})

# Converting the county to regions based on https://www.calbhbc.org/region-map-and-listing.html
counties = {
    "Los Angeles County": "Los Angeles",
    "San Diego County": "Southern",
    "Santa Clara County": "Bay Area",
    "Alameda County": "Bay Area",
    "Orange County": "Southern",
    "San Francisco County": "Bay Area",
    "San Mateo County": "Bay Area",
    "Sacramento County": "Central",
    "Santa Barbara County": "Southern",
    "Yolo County": "Central",
    "Monterey County": "Bay Area",
    "Ventura County": "Southern",
    "San Bernardino County": "Southern",
    "Contra Costa County": "Bay Area",
    "Santa Cruz County": "Bay Area",
    "Riverside County": "Southern",
    "Kern County": "Southern",
    "Marin County": "Bay Area",
    "San Luis Obispo County": "Southern",
    "Solano County": "Bay Area",
    "Humboldt County": "Superior",
    "Sonoma County": "Bay Area",
    "Fresno County": "Central",
    "Placer County": "Central",
    "Butte County": "Superior",
    "Shasta County": "Superior",
    "El Dorado County": "Central",
    "Stanislaus County": "Central",
    "San Benito County": "Bay Area",
    "San Joaquin County": "Central",
    "Mendocino County": "Superior",
    "Tuolumne County": "Central",
    "Siskiyou County": "Superior",
    "Trinity County": "Superior",
    "Merced County": "Central",
    "Lake County": "Superior",
    "Napa County": "Bay Area",
    "Imperial County": "Southern",
}

# Feature Extraction
# --------------------------------

# Add County to the dataset then drop Zipcode
df["County"] = df["ZIPCode"].map(dict_zip)
df.drop("ZIPCode", axis=1, inplace=True)

df["Region"] = df["County"].map(counties)

# Create AgeGroup by binning age
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 30, 40, 50, 60, 100],
    labels=["18-30", "31-40", "41-50", "51-60", "60-100"],
)

# Create Income Class by binning Income
df["IncomeGroup"] = pd.cut(
    x=df["Income"],
    bins=[0, 50, 140, 224],
    labels=["Lower", "Middle", "Upper"],
)

# Convert selected columns to categorical variables
cat_columns = [
    "Family",
    "Education",
    "Personal_Loan",
    "Securities_Account",
    "CD_Account",
    "Online",
    "CreditCard",
    "County",
    "Region",
]
df[cat_columns] = df[cat_columns].astype("category")


# Designing the Visuals on the App
# -------------------------------------

# Button that allows the user to see the entire table
check_data = st.toggle("Show the Original Dataset")
if check_data:
    start, end = st.slider(
        "Select number of rows to display", 0, len(data), (0, 5))
    st.dataframe(data.iloc[start:end])

# Features Creation
# ---------------------------------
st.subheader("Feature Creation", divider="green")
geo_cont = st.container()
with geo_cont:
    # Create layout columns
    col1, col2 = st.columns(2)

    # Display the Counties created
    with col1:
        with st.expander("##### Converted `Zip Codes` to `County`"):
            st.dataframe(df["County"].value_counts(), width=300)

    # Display the Regions created
    with col2:
        with st.expander("##### Created `Region` from `Counties`"):
            st.dataframe(df["Region"].value_counts(), width=300)

agegroup_cont = st.container()
with agegroup_cont:
    # Create more layout columns
    col3, col4 = st.columns(2)

    # Display the Age Groups created
    with col3:
        with st.expander("##### Created `AgeGroup` from `Age`"):
            st.dataframe(df["AgeGroup"].value_counts(), width=300)

    # Display the Income Groups created
    with col4:
        with st.expander("##### Created `Income Group` from `Income`"):
            st.dataframe(df["IncomeGroup"].value_counts(), width=300)


# Button that allows the user to see the entire table
check_df = st.toggle("Show the New Dataset")
if check_df:
    start, end = st.slider(
        "Select number of rows to display", 0, len(df), (0, 5))
    st.dataframe(df.iloc[start:end])

st.subheader("Have fun with data exploration!", divider="green")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(
    ["Bar Plot", "Pair Plot", "Box Plot", "Summary Statistics"])

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
non_numeric_columns = df.select_dtypes(exclude=np.number).columns.tolist()
non_numeric_columns.remove("County")

# Tab 1: Bar Plot
with tab1:
    sb1 = tab1.selectbox(
        "Which feature are you interested in?", non_numeric_columns, key="sel1"
    )
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=sb1, ax=ax)
    total = len(df[sb1])
    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2.0,
            height + 0.5,
            f"{100 * height / total:.2f}%",
            ha="center",
            size=10,
        )
    tab1.pyplot()

# Tab 2: Pair Plot
with tab2:
    multi1 = tab2.multiselect(
        "Which features are you interested in?",
        numeric_columns,
        ["Age", "Income", "Mortgage"],
        key="se2",
    )
    # Incase the user makes a mistake by deleting the columns by mistake
    if len(multi1) == 0:
        st.warning(
            "You cannot leave the field empty, Please select one or more columns!"
        )
    else:
        sns.pairplot(df[multi1], palette=["blue", "green"], markers=["o", "s"])
        tab2.pyplot()

# Tab 3: Box Plot
with tab3:
    plt.figure(figsize=(10, 5))
    rd1 = tab3.radio(
        "Select the feature you want to display",
        numeric_columns,
        horizontal=True,
        key="rad1",
    )
    fig = sns.boxplot(data=df, x=rd1, orient="h")
    tab3.pyplot()

# Tab 4: Summary Statistivs
with tab4:
    st.dataframe(data.describe().T, width=800)

st.divider()

# Define available variables for X, Color, and Facet
x_variables = list(df.columns)
x_variables.remove("County")

# Interactive Distribution Plot
# -------------------------------------------------
st.subheader("Interactive Distribution Plot")

col5, col6, col7 = st.columns(3)
with col5:
    # Selectbox for X variable
    x_variable = st.selectbox(
        "**Select Variable:**", x_variables, index=x_variables.index("Income")
    )

with col6:
    # Selectbox for Color variable with a default "None" option
    color_variable = st.selectbox(
        "**Choose Color:**", ["None"] + non_numeric_columns)

with col7:
    # Selectbox for Facet variable with a default "None" option
    facet_variable = st.selectbox(
        "**Choose Subplot:**",
        ["None"] + non_numeric_columns,
        index=non_numeric_columns.index("Personal_Loan") + 1,
    )

# Check if the selected x_variable is numeric
if x_variable in numeric_columns:
    # Create the Altair chart with binning
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            alt.X(f"{x_variable}:Q", bin=alt.Bin(maxbins=30)),
            alt.Y("count()"),
            alt.Tooltip(),
        )
    )
else:
    # Create the Altair chart without binning
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            alt.X(f"{x_variable}"),
            alt.Y("count()"),
        )
    )

# Check if a Color variable is selected
if color_variable != "None":
    chart = chart.encode(alt.Color(f"{color_variable}:N"))
else:
    chart = chart.encode(color=alt.value("gray"))  # Default color

# Check if Facet variable is selected
if facet_variable != "None":
    chart = (
        chart.properties(width=300, height=300)
        .facet(f"{facet_variable}:O", columns=3)
        .resolve_scale(y="independent")
    )
else:
    # Adjust the figure size when only a single plot is displayed (Facet is None)
    chart = chart.properties(
        width=600, height=500
    )  # Adjust the width and height as needed

# Display the Altair chart in the Streamlit app
st.altair_chart(chart)

st.divider()

# Interactive Scatterplot
# -------------------------------------------------
st.subheader("Interactive Scatterplot")
col8, col9, col10, col11 = st.columns(4)
with col8:
    x_dropdown = st.selectbox(
        "**Choose X Variable:**", x_variables, index=x_variables.index("CCAvg")
    )
with col9:
    y_dropdown = st.selectbox(
        "**Choose Y Variable:**", x_variables, index=x_variables.index("Income")
    )
with col10:
    color_dropdown = st.selectbox(
        "**Choose Color:**",
        ["None"] + x_variables,
        index=x_variables.index("Personal_Loan") + 1,
    )
    with col11:
        facet_dropdown = st.selectbox(
            "**Choose Subplot:**", ["None"] + non_numeric_columns
        )

rect_chart = (
    alt.Chart(df)
    .mark_point()
    .encode(
        alt.X(f"{x_dropdown}"),
        alt.Y(f"{y_dropdown}"),
        alt.Color(f"{color_dropdown}"),
        alt.Tooltip(),
    )
)

# Check if a Color variable is selected
if color_dropdown != "None":
    rect_chart = rect_chart.encode(alt.Color(f"{color_dropdown}"))
else:
    rect_chart = rect_chart.encode(color=alt.value("gray"))  # Default color

# Check if Facet variable is selected
if facet_dropdown != "None":
    rect_chart = (
        rect_chart.properties(width=300, height=300)
        .facet(f"{facet_dropdown}:O", columns=3)
        .resolve_scale(x="independent", y="independent")
    )
else:
    # Adjust the figure size when only a single plot is displayed (Facet is None)
    rect_chart = rect_chart.properties(
        width=600, height=500
    )  # Adjust the width and height as needed

# Display the Altair chart in the Streamlit app
st.altair_chart(rect_chart)

st.divider()

# Interesting Findings
# ----------------------------------------
st.subheader("Interesting Findings")

# Plot correlation heatmap
heat_cont = st.container()
with heat_cont:
    st.write("#### Correlation Heatmap")
    corr_heatmap = sns.heatmap(
        data=df.corr(numeric_only=True), linewidths=0.5, annot=True, fmt=".2f"
    )
    st.pyplot()
    with st.expander("See explanation"):
        st.write(
            """
            The chart above shows ...
        """
        )

# Plot Income distribution
income_cont = st.container()
with income_cont:
    st.write("#### Income Distribution")
    sns.distplot(df[df["Personal_Loan"] == 0]["Income"], color="g")
    sns.distplot(df[df["Personal_Loan"] == 1]["Income"], color="r")
    st.pyplot()
    with st.expander("See explanation"):
        st.write(
            """
            The chart above shows ...
        """
        )

# Plot Family stripplot
family_cont = st.container()
with family_cont:
    st.write("#### Income/Family Stripplot")
    ax = sns.stripplot(x="Family", y="Income",
                       hue="Personal_Loan", data=df, dodge=True)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    st.pyplot()
    with st.expander("See explanation"):
        st.write(
            """
            The chart above shows...
        """
        )
