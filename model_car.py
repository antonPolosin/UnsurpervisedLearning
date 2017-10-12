# Task.1.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def data_prep():
	df = pd.read_csv('model_car_sales.csv') #read dataset from model_car_sales.csv
	
	# removed rows with NULL WAGON SEDAN AND HATCH
	mask = df['UTE'].isnull()
	df.loc[mask, 'UTE'] = np.nan
	df = df[np.isfinite(df['UTE'])]
	
	
	df.drop(['REPORT_DATE', 'DEALER_CODE', 'UTE', 'K__SALES_TOT'], axis=1, inplace=True)
	
	return df

def car_displot():
	df = data_prep()
	
	# Distribution of K__SALES_TOT
	#K__SALES_TOT_dist = sns.distplot(df['K__SALES_TOT'].dropna())
	#plt.show()
	
	# Distribution of HATCH
	HATCH_dist = sns.distplot(df['HATCH'].dropna())
	#plt.show()
	
	HATCH_dist = sns.distplot(df['HATCH'].dropna(), bins=100)
	plt.show()
	
	# Distribution of WAGON
	WAGON_dist = sns.distplot(df['WAG0N'].dropna())
	#plt.show()
	
	WAGON_dist = sns.distplot(df['WAG0N'].dropna(), bins=100)
	plt.show()
	
	# Distribution of SEDAN
	SEDAN_dist = sns.distplot(df['SEDAN'].dropna())
	plt.show()
	
def k_cluster_normal():
	df = data_prep()
	
	df2 = df[['HATCH', 'WAG0N', 'SEDAN']]
	scaler = StandardScaler()

	X = df2.as_matrix()
	
	# set the random state. different random state seeds might result in different centroids locations
	model = KMeans(n_clusters=3, random_state=42)
	model.fit(X)

	# sum of intra-cluster distances
	print("Sum of intra-cluster distance:", model.inertia_)

	print("Centroid locations:")
	for centroid in model.cluster_centers_:
		print(centroid)
		
	#visual
	model = KMeans(n_clusters=3, random_state=42).fit(X)
	
	# assign cluster ID to each record in X
	# Ignore the warning, does not apply to our case here
	y = model.predict(X)
	df2['Cluster_ID'] = y

	# how many records are in each cluster
	print("Cluster membership")
	print(df2['Cluster_ID'].value_counts())

	# pairplot the cluster distribution.
	cluster_g = sns.pairplot(df2, hue='Cluster_ID')
	plt.show()
	
def k_cluster_standard():
	df = data_prep()
	
	df2 = df[['HATCH', 'WAG0N', 'SEDAN']]
	scaler = StandardScaler()

	X = df2.as_matrix()
	X = scaler.fit_transform(X)
	
	# set the random state. different random state seeds might result in different centroids locations
	model = KMeans(n_clusters=3, random_state=42)
	model.fit(X)

	# sum of intra-cluster distances
	print("Sum of intra-cluster distance:", model.inertia_)

	print("Centroid locations:")
	for centroid in model.cluster_centers_:
		print(centroid)
		
	#visual
	model = KMeans(n_clusters=3, random_state=42).fit(X)
	
	# assign cluster ID to each record in X
	# Ignore the warning, does not apply to our case here
	y = model.predict(X)
	df2['Cluster_ID'] = y

	# how many records are in each cluster
	print("Cluster membership")
	print(df2['Cluster_ID'].value_counts())

	# pairplot the cluster distribution.
	cluster_g = sns.pairplot(df2, hue='Cluster_ID')
	plt.show()
	
def cluster_distplot():
	df = data_prep()
	
	df2 = df[['HATCH', 'WAG0N', 'SEDAN']]
	scaler = StandardScaler()

	X = df2.as_matrix()
	X = scaler.fit_transform(X)
	
	# set the random state. different random state seeds might result in different centroids locations
	model = KMeans(n_clusters=3, random_state=42)
	model.fit(X)
	
	model = KMeans(n_clusters=3, random_state=42).fit(X)
	
	# assign cluster ID to each record in X
	# Ignore the warning, does not apply to our case here
	y = model.predict(X)
	df2['Cluster_ID'] = y
	
	# prepare the column and bin size. Increase bin size to be more specific, but 20 is more than enough
	cols = ['HATCH', 'WAG0N', 'SEDAN']
	n_bins = 20

	# inspecting cluster 0
	print("Distribution for cluster 0")
	cluster_to_inspect = 0

	# create subplots
	fig, ax = plt.subplots(nrows=3)
	ax[0].set_title("Cluster {}".format(cluster_to_inspect))

	for j, col in enumerate(cols):
		# create the bins
		bins = np.linspace(min(df2[col]), max(df2[col]), 20)
		# plot distribution of the cluster using histogram
		sns.distplot(df2[df2['Cluster_ID'] == cluster_to_inspect][col], bins=bins, ax=ax[j], norm_hist=True)
		# plot the normal distribution with a black line
		sns.distplot(df2[col], bins=bins, ax=ax[j], hist=False, color="k")

	plt.tight_layout()
	plt.show()

	# inspecting cluster 1
	print("Distribution for cluster 1")
	cluster_to_inspect = 1

	# again, subplots
	fig, ax = plt.subplots(nrows=3)
	ax[0].set_title("Cluster {}".format(cluster_to_inspect))

	for j, col in enumerate(cols):
		# create the bins
		bins = np.linspace(min(df2[col]), max(df2[col]), 20)
		# plot distribution of the cluster using histogram
		sns.distplot(df2[df2['Cluster_ID'] == cluster_to_inspect][col], bins=bins, ax=ax[j], norm_hist=True)
		# plot the normal distribution with a black line
		sns.distplot(df2[col], bins=bins, ax=ax[j], hist=False, color="k")
		
	plt.tight_layout()
	plt.show()
	
def elbow():
	df = data_prep()
	
	df2 = df[['HATCH', 'WAG0N', 'SEDAN']]
	scaler = StandardScaler()

	X = df2.as_matrix()
	X = scaler.fit_transform(X)
	
	# list to save the clusters and cost
	clusters = []
	inertia_vals = []

	# this whole process should take a while
	for k in range(2, 15, 2):
		# train clustering with the specified K
		model = KMeans(n_clusters=k, random_state=42, n_jobs=10)
		model.fit(X)
		
		# append model to cluster list
		clusters.append(model)
		inertia_vals.append(model.inertia_)
		
	# plot the inertia
	plt.plot(range(2,15,2), inertia_vals, marker='*')
	plt.show()
	
	print(clusters[1])
	print("Silhouette score for k=4", silhouette_score(X, clusters[1].predict(X)))

	print(clusters[2])
	print("Silhouette score for k=6", silhouette_score(X, clusters[2].predict(X)))