# Task.2.
import pandas as pd
import numpy as np
from apyori import apriori


def data_prep():
	df = pd.read_csv('web_log_data.csv')
	
	#Drop 'date_time', 'ip', 'step', 'session'. Though there is no need for this as in our research. We need only 2 variables 'request' and 'user_id'
	df.drop(['date_time', 'ip', 'step', 'session'], axis=1, inplace=True)
	
	#full list of items in the "Product_Name" variable
	#df['Product_Name'].value_counts()
	return df
	
def apyori():
	df = data_prep()
	
	# group by account, then list all services
	requests = df.groupby(['user_id'])['request'].apply(list)

	print(requests.head(200))
	
	from apyori import apriori

	# type cast the transactions from pandas into normal list format and run apriori
    # change min support level for more interesting rules
	request_list = list(requests)
	results = list(apriori(request_list, min_support=0.04)) #0.01 or 0.03

	# print first 5 rules
	print(results[:5])
	
	return results
	
def result():
	results = apyori()
	
	result_df = convert_apriori_results_to_pandas_df(results)

	print(result_df.head(50))
	
	result_df = result_df.sort_values(by='Lift', ascending=False)
	print(result_df.head(10))
	
def convert_apriori_results_to_pandas_df(results):
    rules = []
    
    for rule_set in results:
        for rule in rule_set.ordered_statistics:
            rules.append([','.join(rule.items_base), ','.join(rule.items_add), # items_base = left side of rules, items_add = right side
                         rule_set.support, rule.confidence, rule.lift]) # support, confidence and lift for respective rules
    
    return pd.DataFrame(rules, columns=['Left_side', 'Right_side', 'Support', 'Confidence', 'Lift']) # typecast it to pandas df