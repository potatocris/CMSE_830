# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np

# Libraries to help with data visualization
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import altair as alt
import hiplot as hip
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)
# st.set_page_config(layout="wide")

# setting the layout for the seaborn plot
sns.set(style="darkgrid")

# I want help a bank identify customers likely to accept personal loan offers, 
# and ultimately drive growth and profitability

# Load the dataset
file = r'Loan_Modelling.csv'
df = pd.read_csv(file, index_col="ID")

# Set Streamlit app title
st.title("Enhancing AllLife Bank's Personal Loan Marketing Strategy 🏦")

# AllLife Bank has a growing customer base. Majority of these customers are liability customers (depositors) with varying size of deposits. 
# The number of customers who are also borrowers (asset customers) is quite small, and the bank is interested in expanding this base rapidly 
# to bring in more loan business and in the process, earn more through the interest on loans. In particular, the management wants to explore ways 
# of converting its liability customers to personal loan customers (while retaining them as depositors).
# A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9% success. 
# This has encouraged the retail marketing department to devise campaigns with better target marketing to increase the success ratio with a minimal budget.

# Add an expander
with st.expander("**Background & Context**"):
    st.write(
    """
    AllLife Bank aims to grow its customer base, focusing on increasing the number of borrowers (asset customers) while retaining 
    depositors (liability customers). Last year's campaign for liability customers had a conversion rate of over 9%, inspiring 
    the retail marketing department to create more efficient, targeted campaigns with a minimal budget to boost this ratio further.
    """
    )


with st.expander("**Data Dictionary**"):
    st.write(
    """
         * `ID`: Unique Customer Identification Number
         * `Age`: Customer’s age in years
         * `Experience`: Years of professional experience
         * `Income`: Annual income of the customer (in thousand dollars)
         * `ZIP Code`: Home Address ZIP code.
         * `Family`: Family size of the customer
         * `CCAvg`: Avg. spending on credit cards per month (in thousand dollars)
         * `Education`: Education Level. 1: Undergrad; 2: Graduate;3: Advanced/Professional
         * `Mortgage`: Value of house mortgage if any. (in thousand dollars)
         * `Personal_Loan`: Did this customer accept the personal loan offered in the last campaign?
         * `Securities_Account`: Does the customer have securities account with the bank?
         * `CD_Account`: Does the customer have a certificate of deposit (CD) account with the bank?
         * `Online`: Do customers use internet banking facilities?
         * `CreditCard`: Does the customer use a credit card issued by Universal Bank?
         """
         )


# Designing the Visuals on the App
# --------------------------------

# Partitioning the Web App to accommodate the Visualization of the Dataset and ML algorithm
st.sidebar.write("### This Application is divided into two sections")

main_opt = st.sidebar.radio('What do you want to do: ', 
                            ["Data Visualization", "Run Machine Learning Algorithms"])

if(main_opt == "Data Visualization"):
    
    

    # Create layout columns
    col1, col2 = st.columns([1, 3])
    
    # Display unique value counts for each column
    col1.subheader("Value Count", divider="blue")
    col1.write(f"Dataset shape: {df.shape}")
    unique_counts = df.nunique().sort_values(ascending=False)
    col1.write(unique_counts)

    # Convert selected columns to categorical variables
    cat_columns = [
        "Family",
        "Education",
        "Personal_Loan",
        "Securities_Account",
        "CD_Account",
        "Online",
        "CreditCard"
        ]
    df[cat_columns] = df[cat_columns].astype("category")
    
    # Summary statistics
    col2.subheader("Summary Statistics", divider="red")
    col2.write(df.describe().T)

    # Handle negative values in 'Experience'
    neg_experience_count = (df["Experience"] < 0).sum()
    col2.write(
        f"{neg_experience_count} records have negative values for years of experience."
        )
    df["Experience"] = df["Experience"].clip(lower=0)

    # Impute missing values in 'Experience' based on median experience for each age group
    df["Experience"] = df.groupby("Age")["Experience"].transform(
        lambda x: x.fillna(x.median())
        )


    # Button that allows the user to see the entire table
    check_data = st.checkbox('Show the Dataset')
    if check_data:
        st.dataframe(df)

    st.write("### Have fun with data exploration!")

    # Create tabs for different visualizations
    sec1, sec2, sec3, sec4, sec5 = st.tabs(["Variable Distribution", 
                                            "Boxplot", 
                                            "Pair Plot", 
                                            "Bar Plot", 
                                            "Altair Interactive Plot"
                                            ])

    # Tab 1: Distribution Plot
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    non_numeric_columns = df.select_dtypes(exclude=np.number).columns.tolist()
    selected = sec1.radio("Select the feature you want to display", numeric_columns, horizontal= True, key= "rad1")
    fig = sns.histplot(data=df, x=selected, hue="Personal_Loan")
    sec1.pyplot()

    # Tab 2: Box Plot
    selected2 = sec2.radio("Select the feature you want to display", numeric_columns, horizontal= True, key= "rad2")

    fig = sns.boxplot(data=df, x=selected2, orient="h")
    sec2.pyplot()

    # Tab 3: Pair Plot
    selected3 = sec3.multiselect(
        "Which features are you interested in?",
        [d for d in numeric_columns if d != "Personal_Loan"],
        ["Age", "Income", "Mortgage"],
        key="se3"
        )
    # Incase the user makes a mistake by deleting the columns by mistake
    if (len(selected3) == 0):
        st.write("You cannot leave the field empty, Please select one or more columns!")
    else:
        sns.pairplot(
        df[["Personal_Loan"] + selected3],
        hue="Personal_Loan",
        palette=["blue", "green"],
        markers=["o", "s"]
        )
        sec3.pyplot()

    # Tab 4: Bar Plot
    
    selected4 = sec4.selectbox(
        "Which feature are you interested in?", non_numeric_columns, key="se4"
        )
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=selected4, ax=ax)
    total = len(df[selected4])
    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2.0,
            height + 0.5,
            f"{100 * height / total:.2f}%",
            ha="center",
            size=10
            )
    sec4.pyplot()

    # Tab 5: Altair Plot
    opt1, opt2, opt3 = st.columns(3)

    x_sb = opt1.selectbox('x axis: ', numeric_columns)
    y_sb = opt2.selectbox('y axis: ', numeric_columns)
    color = opt3.selectbox('hue: ', non_numeric_columns)

    chart = alt.Chart(df).mark_point().encode(
        alt.X(x_sb, title= f'{x_sb}'),
        alt.Y(y_sb, title=f'{y_sb}'), 
        color=alt.Color(color)).properties(
            width=700,
            height=550
            ).interactive()
    
    sec5.altair_chart(chart)

    

    # Plot correlation heatmap
    st.write("#### Correlation Heatmap")
    corr_heatmap = sns.heatmap(
        data=df.corr(numeric_only=True), 
        linewidths=0.5, 
        annot=True, 
        fmt=".2f"
        )
    st.pyplot()

else:
    st.write("# Coming Soon! 😊")