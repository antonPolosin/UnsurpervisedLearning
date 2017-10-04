# Task.1.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # display plots
import seaborn as sns # generates plots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Task.2.
from apyori import apriori


############################PRE-PROCESS FOR CLUSTERING#########################
# start of script
def data_prep():
    df = pd.read_csv('model_car_sales.csv', na_filter=False)
    print("################## Initial Data #########################")
    ### Describe Data
    df.info()
    ###ute
    df['UTE'].describe()
    ###hatch
    df['HATCH'].describe()
    ###sedan
    df['SEDAN'].describe()
    ###wagon
    df['WAG0N'].describe()
    ###Drop "UTE","DEALER_CODE", "REPORT_DATE"
    df.drop(['UTE', 'DEALER_CODE', 'REPORT_DATE'], axis=1, inplace=True)
    ###check for missing and errornous values
    df['WAG0N'].value_counts()
    ###inplace missing values (delete them) in 'WAGON' variable  and values which have 6.0 and 8.0 as a record

    ###inplace missing values (delete them) in 'HATCH' variable . There are no errornous values

    ###inplace missing values (delete them) in 'SEDAN' variable . There are no errornous values

    ###inplace missing values (delete them) in 'K__SALES_TOT' variable . There are no errornous values

############################PRE-PROCESS ASSOCIATION MINING#########################
def data_prep_association():
    df = pd.read_csv('pos_transaction.csv', na_filter=False)
    ###Drop "Transaction","Transactin date", "Quantity", "Transaction_id". Though there is no need for this as in our research
    ### we won't use these variables.
    df.drop(['Transaction_id', 'Transactin_Date', 'Quantity'], axis=1, inplace=True)
    ###full list of items in the "Product_Name" variable
    df['Product_Name'].value_counts()
def perform_association():
    df = data_prep_association()
    ###groupby location, then list all all products
    transactions = df.groupby(['Location'])['Product_Name'].apply(list)
    ###Generating association rules for 'Location' and 'Product_Name'
    print(transactions.head(5))