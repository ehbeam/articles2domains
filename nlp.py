#!/usr/bin/python

# Preprocess raw full texts in the following steps:
#   (1) Case-folding and punctuation removal
#   (2) Lemmatization with WordNet
#   (3) Consolidation of n-grams from the lexicon

import os, re
import pandas as pd


# Function to load ngrams from lexicon file
def load_ngrams(file="data/text/lexicon_cogneuro.txt"):
	
	ngrams = []
	ngrams += [token.strip().replace("_", " ") for token in open(file, "r").readlines() if "_" in token]
	ngrams = list(set(ngrams))
	ngrams.sort(key = lambda x: x.count(" "), reverse = True)
	
	return ngrams


# Function to load terms from lexicon file
def load_terms(file="data/text/lexicon_cogneuro.txt"):
	
	return [word.strip() for word in open(file, "r").readlines()]


# Function to load data required by NLP tools
def load_nlp_data(path="data/text/lexicon"):

	from nltk.stem import WordNetLemmatizer
	from nltk.corpus import stopwords

	# Load WordNet lemmatizer from NLTK
	lemmatizer = WordNetLemmatizer()

	# Load English stop words from NLTK
	stops = stopwords.words("english")

	# Load n-grams from lexicon
	ngrams = load_ngrams(file="data/text/lexicon_cogneuro.txt")

	# Load terms from lexicon
	lexicon = load_terms(file="data/text/lexicon_cogneuro.txt")

	return lemmatizer, stops, ngrams, lexicon


# Function to lemmatize tokens except select acronyms and names 
def lemmatize(token, lemmatizer):

	# Terms to exclude from lemmatization
	acronyms = ["abc", "aai", "adhd", "aids", "atq", "asam", "asi", "aqc", "asi", "asq", "ax", "axcpt", "axdpx", "bees", "bas", "bdm", "bis", "bisbas", "beq", "brief", "cai", "catbat", "cfq", "deq", "dlmo", "dospert", "dsm", "dsmiv", "dsm5", "ecr", "edi", "eeg", "eei", "ema", "eq", "fmri", "fne", "fss", "grapes", "hrv", "iri", "isi", "ius", "jnd", "leas", "leds", "locs", "poms", "meq", "mctq", "sans", "ippa", "pdd", "pebl", "pbi", "prp", "mspss", "nart", "nartr", "nih", "npu", "nrem", "pas", "panss", "qdf", "rbd", "rem", "rfq", "sam", "saps", "soc", "srs", "srm", "strain", "suds", "teps", "tas", "tesi", "tms", "ug", "upps", "uppsp", "vas", "wais", "wisc", "wiscr", "wrat", "wrat4", "ybocs", "ylsi"]
	names = ["american", "badre", "barratt", "battelle", "bartholomew", "becker", "berkeley", "conners", "corsi", "degroot", "dickman", "marschak", "beckerdegrootmarschak", "beery", "buktenica", "beerybuktenica", "benton", "bickel", "birkbeck", "birmingham", "braille", "brixton", "california", "cambridge", "cattell", "cattells", "chapman", "chapmans", "circadian", "duckworth", "duckworths", "eckblad", "edinburgh", "erickson", "eriksen", "eysenck", "fagerstrom", "fitts", "gioa", "glasgow", "golgi", "gray oral", "halstead", "reitan", "halsteadreitan", "hamilton", "hayling", "holt", "hooper", "hopkins", "horne", "ostberg", "horneostberg", "iowa", "ishihara", "kanizsa", "kaufman", "koechlin", "laury", "leiter", "lennox", "gastaut", "lennoxgastaut", "london", "macarthur", "maudsley", "mcgurk", "minnesota", "montreal", "morris", "mullen", "muller", "lyer", "mullerlyer", "munich", "parkinson", "pavlovian", "peabody", "penn", "penns", "piaget", "piagets", "pittsburgh", "porteus", "posner", "rey", "ostereith", "reyostereith", "reynell", "rivermead", "rutledge", "salthouse", "babcock", "spielberger", "spielbergers", "stanford", "binet", "shaver", "simon", "stanfordbinet", "sternberg", "stroop", "toronto", "trier", "yale", "brown", "umami", "uznadze", "vandenberg", "kuse", "vernier", "vineland", "warrington", "warringtons", "wason", "wechsler", "wisconsin", "yalebrown", "zimbardo", "zuckerman"]
	
	# Lemmatize the token
	if token not in acronyms + names:
		return lemmatizer.lemmatize(token)
	else:
		return token


# Function for stemming, conversion to lowercase, and removal of punctuation
def preprocess(text, lemmatizer, stops):

	# Convert to lowercase, convert slashes to spaces, and remove remaining punctuation except periods
	text = text.replace("-\n", "").replace("\n", " ").replace("\t", " ")
	text = "".join([char for char in text.lower() if char.isalpha() or char.isdigit() or char in [" ", "."]])
	text = text.replace(".", " . ").replace("  ", " ").strip()
	text = re.sub("\. \.+", ".", text)

	# Perform lemmatization, excluding acronyms and names in RDoC matrix
	text = " ".join([lemmatize(token, lemmatizer) for token in text.split() if token not in stops])
	return text


# Function to consolidate n-grams from ontologies
def combine_ngrams(text, ngrams):
	
	# Reverse previous n-gram consolidation, if any
	text = text.replace("_", " ")

	# Perform ngram consolidation, combining spaces with underscores in select n-grams
	for ngram in ngrams:
		text = text.replace(ngram, ngram.replace(" ", "_"))
	text = re.sub("\. \.+", ".", text)
	return text


# Function to compute a document-term matrix (DTM) from texts
def compute_dtm(lexicon, dir="data/text/corpus", outfile="data/text/dtm.csv.gz"):

	from sklearn.feature_extraction.text import CountVectorizer

	# Load preprocessed cogneuro texts
	file_ids = [file.replace(".txt", "") for file in os.listdir(dir) if not file.startswith(".")]
	records = [open("{}/{}.txt".format(dir, file), "r").read() for file in file_ids]

	# Compute DTM with restricted vocabulary
	vec = CountVectorizer(min_df=1, vocabulary=lexicon)
	dtm = vec.fit_transform(records)
	dtm_df = pd.DataFrame(dtm.toarray(), index=list(file_ids), columns=vec.get_feature_names())
	dtm_df = dtm_df.loc[:, (dtm_df != 0).any(axis=0)] # Remove terms with no occurrences
	dtm_df.to_csv(outfile, compression="gzip", index=list(file_ids), columns=dtm_df.columns)

