#!/usr/bin/python

""" 
Input: 		Raw full texts of neuroimaging articles and pre-aggregated brain coordinate data 
Output: 	A set of domains which each entail a list of mental functions and a circuit of brain structures 

Reference:	Beam, E.H., Potts, C., Poldrack, R.A., & Etkin, A. (2021). 
			A data-driven framework for mapping domains of human neurobiology. 
			Nature Neuroscience, 24. DOI: 10.1038/s41593-021-00948-9. 
"""

__author__	  	= "Elizabeth Beam"
__email__ 		= "ebeam@stanford.edu"
__copyright__   = "Copyright 2021, Elizabeth Beam"
__license__	 	= "MIT"
__version__		= "1.0.0"


# Built-in python modules
import argparse
import collections
import os

# Computing and machine learning modules
import numpy as np
import pandas as pd
import sklearn

# Modules in this repository 
import nlp
import ontology

# Arguments for customizing analyses and plots
parser = argparse.ArgumentParser(description="Pipeline for generating data-driven domains of human brain function")
parser.add_argument("--run_all", action="store_true", help="Whether to run the full pipeline: text preprocessing, circuit mapping, term mapping")
parser.add_argument("--run_text_preprocessing", action="store_true", help="Whether to run text preprocessing")
parser.add_argument("--run_domain_mapping", action="store_true", help="Whether to run mapping of domains with circuits and associated terms")
parser.add_argument("--path_texts_in", type=str, default="data/text/raw", help="Directory for raw article full texts")
parser.add_argument("--path_texts_out", type=str, default="data/text/corpus", help="Directory for preprocessed article full texts")
parser.add_argument("--path_results", type=str, default="results", help="Directory for resulting circuit and term lists")
parser.add_argument("--file_lexicon", type=str, default="data/text/lexicon_cogneuro.txt", help="Input file with list of terms to extract from full texts")
parser.add_argument("--file_lexicon_labels", type=str, default="data/text/labels_cogneuro.csv", help="Input file with domain title labels for terms in the lexicon")
parser.add_argument("--file_dtm", type=str, default="data/text/dtm.csv.gz", help="Intermediary file with document-term matrix")
parser.add_argument("--file_coordinates", type=str, default="data/brain/coordinates.csv", help="Input file with brain coordinates")
parser.add_argument("--file_atlas_labels", type=str, default="data/brain/labels.csv", help="Input file with Harvard-Oxford atlas labels")
parser.add_argument("--file_circuits", type=str, default="results/circuits.csv", help="Output file with brain circuit from clusting structures by terms")
parser.add_argument("--file_terms", type=str, default="results/terms.csv", help="Output file with list of terms associated with each circuit")
parser.add_argument("--k", type=int, default=6, help="Number of domains selected for the data-driven ontology")
parser.add_argument("--n_terms", default=range(5,26), help="Range of number of terms per data-driven domain")
args = parser.parse_args()

# Example usage for staged processing:
# 1.	python main.py --run_text_preprocessing --path_texts_in "data/text/raw" --path_texts_out "data/text/corpus" --file_dtm "data/text/dtm.csv.gz"
# 2.	python main.py --run_domain_mapping --k 6 --file_dtm "data/text/dtm.csv.gz" --file_circuits "circuits.csv" --file_terms "terms.csv"

# ETA provided for each stage of processing 3,411 PMC open access articles on a 2019 MacBook Air


################################################
########### 1. Preprocess the texts ############
################################################

# Stage ETA: 20 minutes

if args.run_all or args.run_text_preprocessing:

	print("\n------- PREPROCESSING THE TEXTS -------\n")

	# Instantiate the directory for preprocessed texts
	if not os.path.exists(args.path_texts_out):
		os.makedirs(args.path_texts_out)

	# Load the data for NLP tools
	lemmatizer, stops, ngrams, lexicon = nlp.load_nlp_data()

	# Loop over articles in the directory of raw texts
	files = [file for file in os.listdir(args.path_texts_in) if not file.startswith(".")]
	n = len(files)
	for i, file in enumerate(files): # ETA 18 minutes (~0.3 seconds/file)
		
		infile = "{}/{}".format(args.path_texts_in, file)
		outfile = "{}/{}".format(args.path_texts_out, file)
		
		if os.path.isfile(infile) and not os.path.isfile(outfile): # Skip previously preprocessed texts
			text = open(infile, "r").read() 
			text = nlp.preprocess(text, lemmatizer, stops)
			text = nlp.combine_ngrams(text, ngrams)
			with open(outfile, "w+") as fout:
				fout.write(text)

		if (i % 1000 == 0):
			print("\tText {:6d} / {:6d}".format(i, n))
		if (i == n-1):
			print("\tText {:6d} / {:6d}\n".format(n, n))

	# Compute document-term matrix (DTM) with articles (rows) x terms in lexicon (columns)
	if not os.path.isfile(args.file_dtm): 
		print("\tComputing the document-term matrix\n")
		nlp.compute_dtm(lexicon, dir=args.path_texts_out, outfile=args.file_dtm) # ETA ~1 minute
	else:
		print("\tDocument-term matrix already computed\n")

	print("Completed preprocessing of texts\n")


################################################
############# 2. Map the domains ##############
################################################

# Stage ETA: 20 minutes

if args.run_all or args.run_domain_mapping:

	print("\n------- MAPPING THE DOMAINS -------\n")

	np.random.seed(42)

	# Instantiate the directory for the resulting circuit file
	if not os.path.exists(args.path_results):
		os.makedirs(args.path_results)

	# Load the text and coordinate data
	dtm = ontology.load_doc_term_matrix(file=args.file_dtm, binarize=True)
	act = ontology.load_coordinates(file_coordinates=args.file_coordinates, file_labels=args.file_atlas_labels)

	# Reconcile article IDs
	ids = act.index.intersection(dtm.index)
	dtm = dtm.loc[ids]
	act = act.loc[ids]

	# Split IDs into train, validation, and test sets
	train, val, test = ontology.split_ids(ids)
	print("\tTraining:\t{:5d} articles ({:.2f}%)".format(len(train), 100*len(train)/len(ids)))
	print("\tValidation:\t{:5d} articles ({:.2f}%)".format(len(val), 100*len(val)/len(ids)))
	print("\tTest:\t\t{:5d} articles ({:.2f}%)\n".format(len(test), 100*len(test)/len(ids)))

	# Compute the PMI-weighted structure-term matrix
	stm = ontology.load_stm(act.loc[train], dtm.loc[train])

	# Cluster structures by PMI-weighted co-occurrences with function terms
	print("\tClustering structures into circuits by PMI of occurrences with terms\n") # ETA <1 minute
	circuits = ontology.cluster_structures(args.k, stm, act.columns)
	circuits.to_csv(args.file_circuits, index=None)

	# Assign function terms to circuits by post-biserial correlation of occurrences
	print("\tAssigning candidate terms by correlation of occurrences with circuits\n") # ETA <1 minute
	lists = ontology.assign_functions(args.k, circuits, act.loc[train], dtm.loc[train], list_lens=args.n_terms)
	lists.to_csv(args.file_terms, index=None) 

	# Select number of function terms by mean of forward and reverse inference classifier performance
	ontology.optimize_list_len(args.k, train, val, list_lens=args.n_terms, 
							   file_terms=args.file_terms, file_dtm=args.file_dtm, 
							   file_coordinates=args.file_coordinates, file_labels=args.file_atlas_labels) # ETA 20 minutes

	# Name domains by the function term with highest degree centrality
	print("\tNaming domains by term with highest degree centrality\n")
	lists, circuits = ontology.load_ontology(file_terms=args.file_terms, file_circuits=args.file_circuits)
	k2name = ontology.name_domains(lists, file_dtm=args.file_dtm, file_labels=args.file_lexicon_labels)
	lists, circuits = ontology.export_ontology(lists, circuits, k2name, file_terms=args.file_terms, file_circuits=args.file_circuits)

	print("Completed mapping of domains\n")

