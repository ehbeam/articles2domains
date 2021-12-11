
import os, sklearn
import pandas as pd
import numpy as np


def load_coordinates(file_coordinates="data/brain/coordinates.csv", file_labels="data/brain/labels.csv"):
	
	atlas_labels = pd.read_csv(file_labels)
	activations = pd.read_csv(file_coordinates, index_col=0)
	activations = activations[atlas_labels["PREPROCESSED"]]

	return activations


def load_lexicon(file="data/text/lexicon_cogneuro.txt", tkn_filter=[]):
	
	lexicon = sorted([token.strip() for token in open(file, "r").readlines()])
	if len(tkn_filter) > 0:
		lexicon = sorted(list(set(lexicon).intersection(tkn_filter)))
	
	return lexicon


def doc_mean_thres(df):

	doc_mean = df.mean()
	df_bin = 1.0 * (df.values > doc_mean.values)
	df_bin = pd.DataFrame(df_bin, columns=df.columns, index=df.index)
	
	return df_bin


def load_doc_term_matrix(file="data/text/dtm.csv.gz", binarize=True):
	
	dtm = pd.read_csv(file, index_col=0)
	
	if binarize:
		dtm = doc_mean_thres(dtm)
	
	return dtm


def load_ontology(file_terms="results/terms.csv", file_circuits="results/circuits.csv"):

	lists = pd.read_csv(file_terms, index_col=None)
	circuits = pd.read_csv(file_circuits, index_col=None)

	return lists, circuits


def split_ids(ids, train_prop=0.7, val_prop=0.2, test_prop=0.1):
	
	from sklearn.model_selection import train_test_split
	
	train, rest = train_test_split(ids, test_size=val_prop+test_prop, random_state=42)
	val, test = train_test_split(rest, test_size=test_prop/(val_prop+test_prop), random_state=42)

	return train, val, test


def observed_over_expected(df):

	# From https://github.com/cgpotts/cs224u/blob/master/vsm.py

	col_totals = df.sum(axis=0)
	total = col_totals.sum()
	row_totals = df.sum(axis=1)
	expected = np.outer(row_totals, col_totals) / total
	oe = df / expected

	return oe


def pmi(df, positive=True):

	# From https://github.com/cgpotts/cs224u/blob/master/vsm.py

	df = observed_over_expected(df)
	with np.errstate(divide="ignore"):
		df = np.log(df)
	df[np.isinf(df)] = 0.0  # log(0) = 0
	if positive:
		df[df < 0] = 0.0

	return df


def load_stm(act_bin, dtm_bin):

	stm = np.dot(act_bin.transpose(), dtm_bin)
	stm = pd.DataFrame(stm, columns=dtm_bin.columns, index=act_bin.columns)
	stm = pmi(stm, positive=False)
	stm = stm.dropna(axis=1, how="all") # Drop terms with no co-occurrences

	return stm


def cluster_structures(k, stm, structures):

	from sklearn import cluster

	kmeans = cluster.KMeans(n_clusters=k, max_iter=1000, random_state=42)
	kmeans.fit(stm)
	clust = pd.DataFrame({"STRUCTURE": structures, 
						  "CLUSTER": [l+1 for l in list(kmeans.labels_)]})
	clust = clust.sort_values(["CLUSTER", "STRUCTURE"])

	return clust


def assign_functions(k, clust, act, dtm, list_lens=range(5,26)):

	from scipy.stats import pointbiserialr

	lists = pd.DataFrame()
	for i in range(k):

		structures = list(clust.loc[clust["CLUSTER"] == i+1, "STRUCTURE"])
		centroid = np.mean(act[structures], axis=1)
		lexicon = dtm.columns
		
		R = pd.Series([pointbiserialr(dtm[word], centroid)[0] for word in lexicon], index=lexicon)
		R = R[R > 0].sort_values(ascending=False)[:max(list_lens)]
		R = pd.DataFrame({"CLUSTER": [i+1 for l in range(max(list_lens))], 
						  "TOKEN": R.index, "R": R.values})
		lists = lists.append(R)

	return lists
 

def optimize_hyperparameters(param_list, train_set, val_set, max_iter=100):

	from sklearn.multiclass import OneVsRestClassifier
	from sklearn.linear_model import LogisticRegression	
	from sklearn.metrics import roc_auc_score

	op_score_val, op_fit = 0, 0
  
	for params in param_list:
		
		# Specify the classifier with the current hyperparameter combination
		classifier = OneVsRestClassifier(LogisticRegression(penalty=params["penalty"], C=params["C"], fit_intercept=params["fit_intercept"], 
								max_iter=max_iter, tol=1e-10, solver="liblinear", random_state=42))

		# Fit the classifier on the training set
		classifier.fit(train_set[0], train_set[1])

		# Evaluate on the validation set
		preds_val = classifier.predict_proba(val_set[0])
		if preds_val.shape[1] == 2 and val_set[1].shape[1] == 1: # In case there is only one class
			preds_val = preds_val[:,1] # The second column is for the label 1
		score_val = roc_auc_score(val_set[1], preds_val, average="macro")
		
		# Update outputs if this model is the best so far
		if score_val > op_score_val:
			op_score_val = score_val
			op_fit = classifier

	return op_score_val


def optimize_list_len(k, train, val, list_lens=range(5, 26), 
					 file_terms="results/terms.csv", file_dtm="data/text/dtm.csv.gz", file_circuits="results/circuits.csv", 
					 file_coordinates="data/brain/coordinates.csv", file_labels="data/brain/labels.csv"):

	from sklearn.model_selection import ParameterSampler

	act_bin = load_coordinates()
	dtm_bin = load_doc_term_matrix(file=file_dtm, binarize=True)

	lists, circuits = load_ontology(file_terms=file_terms, file_circuits=file_circuits)
	
	# Specify the hyperparameters for the randomized grid search
	max_iter = 100
	param_grid = {"penalty": ["l1", "l2"],
				  "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
				  "fit_intercept": [True, False]}
	param_list = list(ParameterSampler(param_grid, n_iter=28, random_state=42))
	
	op_lists = pd.DataFrame()
	for circuit in range(1, k+1):

		print("\n\tSelecting terms for domain {:02d}".format(circuit))
		forward_scores, reverse_scores = [], []
		structures = circuits.loc[circuits["CLUSTER"] == circuit, "STRUCTURE"]

		for list_len in list_lens:
			words = lists.loc[lists["CLUSTER"] == circuit, "TOKEN"][:list_len]

			# Optimize forward inference classifier 
			train_set_f = [dtm_bin.loc[train, words], act_bin.loc[train, structures]]
			val_set_f = [dtm_bin.loc[val, words], act_bin.loc[val, structures]]
			try:
				op_val_f = optimize_hyperparameters(param_list, train_set_f, val_set_f, max_iter=max_iter)
			except:
				op_val_f = 0.0
			forward_scores.append(op_val_f)

			# Optimize reverse inference classifier
			train_set_r = [act_bin.loc[train, structures], dtm_bin.loc[train, words]]
			val_set_r = [act_bin.loc[val, structures], dtm_bin.loc[val, words]]
			try:
				op_val_r = optimize_hyperparameters(param_list, train_set_r, val_set_r, max_iter=max_iter)
			except:
				op_val_r = 0.0
			reverse_scores.append(op_val_r)
		
		scores = [(forward_scores[i] + reverse_scores[i])/2.0 for i in range(len(forward_scores))]
		op_len = list_lens[scores.index(max(scores))]
		print("\tHighest mean ROC-AUC = {:6.4f} for {:02d} terms".format(max(scores), op_len))
		op_df = lists.loc[lists["CLUSTER"] == circuit][:op_len]
		op_df["ROC_AUC"] = max(scores)
		op_lists = op_lists.append(op_df)

	op_lists.to_csv(file_terms, index=None)


def term_degree_centrality(i, lists, dtm, ids, reweight=False, term_col="TOKEN"):

	terms = list(set(lists.loc[lists["CLUSTER"] == i, term_col]))
	ttm = np.matmul(dtm.loc[ids, terms].values.T, dtm.loc[ids, terms].values)
	ttm = pd.DataFrame(ttm, index=terms, columns=terms)
	adj = pd.DataFrame(0, index=terms, columns=terms)

	for term_i in terms:
		for term_j in terms:
			adj.loc[term_i, term_j] = ttm.loc[term_i, term_j]

	degrees = adj.sum(axis=1)
	degrees = degrees.loc[terms]
	degrees = degrees.sort_values(ascending=False)

	return degrees


def nounify(word, form2noun):
	if word in form2noun.keys():
		return form2noun[word]
	else: 
		return word


def name_domains(lists, file_dtm="data/text/dtm.csv.gz", file_labels="data/text/labels_cogneuro.csv"):
	
	k = len(set(lists["CLUSTER"]))
	k2terms = {i: list(set(lists.loc[lists["CLUSTER"] == i+1, "TOKEN"])) for i in range(k)}
	k2name = {i+1: "" for i in range(k)}
	names, degs = [""]*k, [0]*k

	dtm_bin = load_doc_term_matrix(file=file_dtm, binarize=True)
	
	while "" in names:
		for i in range(k):
			
			degrees = term_degree_centrality(i+1, lists, dtm_bin, dtm_bin.index)
			degrees = degrees.loc[k2terms[i]].sort_values(ascending=False)
			name = degrees.index[0].upper()
			
			if name not in names:
				names[i] = name
				degs[i] = max(degrees)
				k2name[i+1] = name
			
			elif name in names:
				name_idx = names.index(name)
				if degs[name_idx] > degs[i]:
					k2terms[i] = [term for term in k2terms[i] if term != name.lower()]
	
	title_df = pd.read_csv(file_labels, index_col=None, header=0)
	term2title = {term.upper(): title.upper().replace(" ", "_") for term, title in zip(title_df["TERM"], title_df["TITLE"])}

	k2name = {k: nounify(name, term2title) for k, name in k2name.items()}
	
	return k2name


def export_ontology(lists, circuits, k2name, file_terms="results/terms.csv", file_circuits="results/circuits.csv"):

	lists["DOMAIN"] = [k2name[k] for k in lists["CLUSTER"]]
	lists = lists.sort_values(["CLUSTER", "R"], ascending=[True, False])
	lists = lists[["CLUSTER", "DOMAIN", "TOKEN", "R", "ROC_AUC"]]
	lists.to_csv(file_terms, index=None)

	circuits["DOMAIN"] = [k2name[k] for k in circuits["CLUSTER"]]
	circuits = circuits.sort_values(["CLUSTER", "STRUCTURE"])
	circuits = circuits[["CLUSTER", "DOMAIN", "STRUCTURE"]]
	circuits.to_csv(file_circuits, index=None)

	return lists, circuits


def load_optimized_lists(doms, lists, list_lens, seed_df, vsm):

	from scipy.spatial.distance import cosine

	ops = []
	op_df = pd.DataFrame(index=doms, columns=list_lens)
	for dom in doms:
		seed_tkns = seed_df.loc[seed_df["DOMAIN"] == dom, "TOKEN"]
		seed_tkns = [tkn for tkn in seed_tkns if tkn in vsm.index]
		seed_centroid = np.mean(vsm.loc[seed_tkns])
		for list_len in list_lens:
			len_tkns = lists.loc[lists["DOMAIN"] == dom, "TOKEN"][:list_len]
			len_centroid = np.mean(vsm.loc[len_tkns])
			op_df.loc[dom, list_len] = 1 - cosine(seed_centroid, len_centroid)
		sims = list(op_df.loc[dom])
		idx = sims.index(max(sims))
		ops.append(np.array(list_lens)[idx])
	op_df["OPTIMAL"] = ops
	
	return op_df


def update_lists(doms, op_df, lists, framework, path=""):

	columns = ["ORDER", "DOMAIN", "TOKEN", "SOURCE", "DISTANCE"]
	new = pd.DataFrame(columns=columns)
	for order, dom in enumerate(doms):
		list_len = op_df.loc[dom, "OPTIMAL"]
		dom_df = lists.loc[lists["DOMAIN"] == dom][:list_len]
		new = new.append(dom_df)
	new.to_csv("{}lists/lists_{}_opsim.csv".format(path, framework), index=None)

	return new