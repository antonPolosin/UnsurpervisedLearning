# Task.2.
import pandas as pd
import numpy as np
from apyori import apriori
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt


def data_prep():
	df = pd.read_csv('pos_transactions.csv')
	
	#Drop "LOCATION","Transactin date", "Quantity", "Transaction_id". Though there is no need for this as in our research
	df.drop(['Location', 'Transactin_Date', 'Quantity'], axis=1, inplace=True)
	
	#full list of items in the "Product_Name" variable
	#df['Product_Name'].value_counts()
	return df
	
def apyori():
	df = data_prep()
	
	# group by account, then list all services
	transactions = df.groupby(['Transaction_Id'])['Product_Name'].apply(list)

	print(transactions.head(20))
	
	from apyori import apriori

	# type cast the transactions from pandas into normal list format and run apriori
	transaction_list = list(transactions)
	results = list(apriori(transaction_list, min_support=0.01)) #0.01 or 0.03

	# print first 5 rules
	print(results[:5])
	
	return results
	
def result():
	results = apyori()
	
	result_df = convert_apriori_results_to_pandas_df(results)

	print(result_df.head(20))
	
	result_df = result_df.sort_values(by='Lift', ascending=False)
	print(result_df.head(10))
	return result_df
	
def pairplot():	
	results_df = result()
	df2 = results_df[['Support', 'Confidence', 'Lift']]
	scaler = StandardScaler()
	
	X = df2.as_matrix()
	X = scaler.fit_transform(X)
	
	model = KMeans(n_clusters=3, random_state=42).fit(X)
	y = model.predict(X)
	df2['Cluster_ID'] = y
	
	# how many records are in each cluster
	print("Cluster membership")
	print(df2['Cluster_ID'].value_counts())

	# pairplot the cluster distribution.
	cluster_g = sns.pairplot(df2, hue='Cluster_ID')
	plt.show()
	
def convert_apriori_results_to_pandas_df(results):
    rules = []
    
    for rule_set in results:
        for rule in rule_set.ordered_statistics:
            rules.append([','.join(rule.items_base), ','.join(rule.items_add), # items_base = left side of rules, items_add = right side
                         rule_set.support, rule.confidence, rule.lift]) # support, confidence and lift for respective rules
    
    return pd.DataFrame(rules, columns=['Left_side', 'Right_side', 'Support', 'Confidence', 'Lift']) # typecast it to pandas df