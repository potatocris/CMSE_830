import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

# Set Streamlit app title
st.title("Loan Modelling Data Exploration")

# Add an expander
expand = st.expander(
    """#### Background & Context 
            Click to expand:"""
)
expand.write(
    """AllLife Bank has a growing customer base. Majority of these customers are liability customers (depositors) with varying size of deposits. 
         The number of customers who are also borrowers (asset customers) is quite small, and the bank is interested in expanding this base rapidly 
         to bring in more loan business and in the process, earn more through the interest on loans. In particular, the management wants to explore ways 
         of converting its liability customers to personal loan customers (while retaining them as depositors).
         A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9% success. 
         This has encouraged the retail marketing department t o devise campaigns with better target marketing to increase the success ratio with a minimal budget."""
)

# Add an expander
expand2 = st.expander(
    """#### Data Dictionary 
            Click to expand:"""
)
expand2.write(
    """
         * ID: Customer ID
         * Age: Customer’s age in completed years
         * Experience: #years of professional experience
         * Income: Annual income of the customer (in thousand dollars)
         * ZIP Code: Home Address ZIP code.
         * Family: the Family size of the customer
         * CCAvg: Avg. spending on credit cards per month (in thousand dollars)
         * Education: Education Level. 1: Undergrad; 2: Graduate;3: Advanced/Professional
         * Mortgage: Value of house mortgage if any. (in thousand dollars)
         * Personal_Loan: Did this customer accept the personal loan offered in the last campaign?
         * Securities_Account: Does the customer have securities account with the bank?
         * CD_Account: Does the customer have a certificate of deposit (CD) account with the bank?
         * Online: Do customers use internet banking facilities?
         * CreditCard: Does the customer use a credit card issued by Universal Bank?"""
)

# Load the dataset
df = pd.read_csv('HW4/Loan_Modelling.csv', index_col=0)

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
    "CreditCard",
]
df[cat_columns] = df[cat_columns].astype("category")

# Display descriptive statistics
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

# Plot correlation heatmap
st.write("#### Correlation Heatmap")
corr_heatmap = sns.heatmap(
    data=df.corr(numeric_only=True), linewidths=0.5, annot=True, fmt=".2f"
)
st.pyplot()

st.write("## Have fun with data exploration!")

# Create tabs for different visualizations
sec1, sec2, sec3, sec4 = st.tabs(["Distribution", "Boxplot", "Pair Plot", "Bar Plot"])

# Tab 1: Distribution Plot
sec1.write("### Distribution Plot")
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
selected = sec1.selectbox(
    "Which features are you interested in?",
    [d for d in numeric_columns if d != "Personal_Loan"],
    key="se1",
)
fig = sns.histplot(data=df, x=selected, hue="Personal_Loan", element="step")
sec1.pyplot()

# Tab 2: Box Plot
sec2.write("### Box Plot")
selected2 = sec2.selectbox(
    "Which features are you interested in?", numeric_columns, key="se2"
)
fig = sns.boxplot(data=df, x=selected2, hue="Personal_Loan", orient="h")
sec2.pyplot()

# Tab 3: Pair Plot
sec3.write("### Pair Plot")
selected3 = sec3.multiselect(
    "Which features are you interested in?",
    [d for d in numeric_columns if d != "Personal_Loan"],
    ["Age", "Income", "Mortgage"],
    key="se3",
)
sns.pairplot(
    df[["Personal_Loan"] + selected3],
    hue="Personal_Loan",
    palette=["blue", "green"],
    markers=["o", "s"],
)
sec3.pyplot()

# Tab 4: Bar Plot
sec4.write("### Bar Plot")
non_numeric_columns = df.select_dtypes(exclude=np.number).columns.tolist()
selected4 = sec4.selectbox(
    "Which features are you interested in?", non_numeric_columns, key="se4"
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
        size=10,
    )
sec4.pyplot()


# Add a file uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)
    st.write("uploaded_csv")
    st.write(df_uploaded)


# Add a progress bar
progress = col2.progress(0)
for i in range(100):
    progress.progress(i + 1)

# Add a success box
if progress:
    col2.success(
        "Negative values replaced with median **experience** for each age group!"
    )
