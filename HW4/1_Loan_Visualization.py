# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np
import zipcodes as zcode # to get zipcodes

# Libraries to help with data visualization
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import altair as alt
import hiplot as hip
import streamlit as st

st.set_page_config(
    page_title="Loan Marketing",
    page_icon="🏦",
)
st.set_option('deprecation.showPyplotGlobalUse', False)
# st.set_page_config(layout="wide")

# setting the layout for the seaborn plot
sns.set(style="darkgrid")

# I want help a bank identify customers likely to accept personal loan offers, and ultimately drive growth and profitability

file = r'Loan_Modelling.csv'

# Load the dataset
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, index_col="ID")

    return df

# Load the data using the defined function
df = load_data(file)

# Set Streamlit app title
st.title(":green[Enhancing AllLife Bank's Personal Loan Marketing Strategy] 🏦")

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

data_load_state = st.text('Loading data...')
# Data Preprocessing
# --------------------------------
# The minumum value of Experience column is -3.0 which is a mistake because Year can not be negative. 
# This has to be fixed
# Handle negative values in 'Experience'
df["Experience"] = df["Experience"].clip(lower=0) # Replace negative values in 'Experience' with NaN
df["Experience"] = df.groupby("Age")["Experience"].transform(
    lambda x: x.fillna(x.median())) # Impute missing values in 'Experience' based on median experience for each age group

# Converting Zipcode to County
list_zipcode=df.ZIPCode.unique()
dict_zip={}
for zipcode in list_zipcode:
    city_county = zcode.matching(zipcode.astype('str'))
    if len(city_county)==1:
        county=city_county[0].get('county')
    else:
        county=zipcode
    
    dict_zip.update({zipcode:county})
dict_zip.update({92717:'Orange County'})
dict_zip.update({92634:'Orange County'})
dict_zip.update({96651:'El Dorado County'})
dict_zip.update({93077:'Ventura County'})

# Converting the county to regions based on https://www.calbhbc.org/region-map-and-listing.html
counties = {
'Los Angeles County':'Los Angeles',
'San Diego County':'Southern',
'Santa Clara County':'Bay Area',
'Alameda County':'Bay Area',
'Orange County':'Southern',
'San Francisco County':'Bay Area',
'San Mateo County':'Bay Area',
'Sacramento County':'Central',
'Santa Barbara County':'Southern',
'Yolo County':'Central',
'Monterey County':'Bay Area',            
'Ventura County':'Southern',             
'San Bernardino County':'Southern',       
'Contra Costa County':'Bay Area',        
'Santa Cruz County':'Bay Area',           
'Riverside County':'Southern',            
'Kern County':'Southern',                 
'Marin County':'Bay Area',                
'San Luis Obispo County':'Southern',     
'Solano County':'Bay Area',              
'Humboldt County':'Superior',            
'Sonoma County':'Bay Area',                
'Fresno County':'Central',               
'Placer County':'Central',                
'Butte County':'Superior',               
'Shasta County':'Superior',                
'El Dorado County':'Central',             
'Stanislaus County':'Central',            
'San Benito County':'Bay Area',          
'San Joaquin County':'Central',           
'Mendocino County':'Superior',             
'Tuolumne County':'Central',                
'Siskiyou County':'Superior',              
'Trinity County':'Superior',                
'Merced County':'Central',                  
'Lake County':'Superior',                 
'Napa County':'Bay Area',                   
'Imperial County':'Southern',
}

# Feature Extraction
# --------------------------------

# Add County to the dataset then drop Zipcode
df['County']=df['ZIPCode'].map(dict_zip) 
df.drop("ZIPCode", axis=1, inplace=True)
st.dataframe(df["County"].value_counts() , width=300)

df['Regions'] = df['County'].map(counties)
st.dataframe(df["Regions"].value_counts() , width=300)

# Create Agebin by binning age
df['Agebin'] = pd.cut(df['Age'], bins = [0, 30, 40, 50, 60, 100], 
                           labels = ['18-30', '31-40', '41-50', '51-60', '60-100'])
st.dataframe(df["Agebin"].value_counts() , width=300)

# Create Income Class by binning Income
df["Income_Group"] = pd.cut(
    x=df["Income"],
    bins=[0, 50, 140, 224],
    labels=["Lower", "Middle", "Upper"],
)
st.dataframe(df["Income_Group"].value_counts() , width=300)

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

    
    # Summary statistics
    col2.subheader("Summary Statistics", divider="red")
    col2.write(df.describe().T)

    # Button that allows the user to see the entire table
    check_data = st.toggle('Show the Dataset')
    if check_data:
        st.dataframe(df)
    
    st.divider()

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
    non_numeric_columns.remove("County")

    selected = sec1.radio("Select the feature you want to display", numeric_columns, horizontal= True, key= "rad1")
    fig = sns.histplot(data=df, x=selected, hue="Personal_Loan")
    sec1.pyplot()

    # Tab 2: Box Plot
    selected2 = sec2.radio("Select the feature you want to display", numeric_columns, horizontal= True, key= "rad2")

    fig = sns.boxplot(data=df, x=selected2, orient="h")
    sec2.pyplot()

    # # Tab 3: Pair Plot
    # selected3 = sec3.multiselect(
    #     "Which features are you interested in?",
    #     [d for d in numeric_columns if d != "Personal_Loan"],
    #     ["Age", "Income", "Mortgage"],
    #     key="se3"
    #     )
    # # Incase the user makes a mistake by deleting the columns by mistake
    # if (len(selected3) == 0):
    #     st.write("You cannot leave the field empty, Please select one or more columns!")
    # else:
    #     sns.pairplot(
    #     df[["Personal_Loan"] + selected3],
    #     hue="Personal_Loan",
    #     palette=["blue", "green"],
    #     markers=["o", "s"]
    #     )
    #     sec3.pyplot()

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

    

    # # Plot correlation heatmap
    # st.write("#### Correlation Heatmap")
    # corr_heatmap = sns.heatmap(
    #     data=df.corr(numeric_only=True), 
    #     linewidths=0.5, 
    #     annot=True, 
    #     fmt=".2f"
    #     )
    # st.pyplot()


    # sns.distplot( df[df['Personal_Loan'] == 0]['Income'], color = 'g')
    # sns.distplot( df[df['Personal_Loan'] == 1]['Income'], color = 'r')
    # st.pyplot()

    # ax = sns.stripplot(x='Family', y='Income', hue='Personal_Loan', data=df,dodge= True)
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    # st.pyplot()

    from vega_datasets import data

    # We use @st.cache_data to keep the dataset in cache
    @st.cache_data
    def get_data():
        source = data.stocks()
        source = source[source.date.gt("2004-01-01")]
        return source

    source = get_data()

    # Define the base time-series chart.
    def get_chart(data):
        hover = alt.selection_single(
            fields=["date"],
            nearest=True,
            on="mouseover",
            empty="none",
        )

        lines = (
            alt.Chart(data, title="Evolution of stock prices")
            .mark_line()
            .encode(
                x="date",
                y="price",
                color="symbol",
            )
        )

        # Draw points on the line, and highlight based on selection
        points = lines.transform_filter(hover).mark_circle(size=65)

        # Draw a rule at the location of the selection
        tooltips = (
            alt.Chart(data)
            .mark_rule()
            .encode(
                x="yearmonthdate(date)",
                y="price",
                opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
                tooltip=[
                    alt.Tooltip("date", title="Date"),
                    alt.Tooltip("price", title="Price (USD)"),
                ],
            )
            .add_selection(hover)
        )
        return (lines + points + tooltips).interactive()

    chart = get_chart(source)

    # Add annotations
    ANNOTATIONS = [
        ("Mar 01, 2008", "Pretty good day for GOOG"),
        ("Dec 01, 2007", "Something's going wrong for GOOG & AAPL"),
        ("Nov 01, 2008", "Market starts again thanks to..."),
        ("Dec 01, 2009", "Small crash for GOOG after..."),
    ]
    annotations_df = pd.DataFrame(ANNOTATIONS, columns=["date", "event"])
    annotations_df.date = pd.to_datetime(annotations_df.date)
    annotations_df["y"] = 10

    annotation_layer = (
        alt.Chart(annotations_df)
        .mark_text(size=20, text="⬇", dx=-8, dy=-10, align="left")
        .encode(
            x="date:T",
            y=alt.Y("y:Q"),
            tooltip=["event"],
        )
        .interactive()
    )

    st.altair_chart((chart + annotation_layer).interactive(),
    use_container_width=True)







    map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

    st.map(map_data)

    
    age = st.slider('How old are you?', 0, 130, 25)
    st.write("I'm ", age, 'years old')

    values = st.slider('Select a range of values', 0.0, 100.0, (25.0, 75.0))
    st.write('Values:', values)





    data_load_state.text('Loading data...done!')


else:
    st.write("# Coming Soon! 😊")