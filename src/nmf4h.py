#NMF with SC
from gensim.test.utils import datapath
from gensim.models import Nmf
from gensim.corpora.dictionary import Dictionary
import pandas as pd
from gensim.test.utils import datapath
from gensim.matutils import sparse2full


import pickle
import numpy as np
import os
import pandas as pd
from collections import Counter

def print_sample_rows(matrix, name, num_rows=3):
    """
    Prints a sample of rows from the given matrix.
    Args:
        matrix: 2D array-like, matrix to sample rows from.
        name: str, name of the matrix to display.
        num_rows: int, number of rows to display.
    """
    print(f"Sample rows from {name}:")
    print(matrix[:num_rows, :])
    print("-" * 50)

def term_weighting(nursing_notes):
    """
    Computes Term-Weighting (TW) for a corpus of nursing notes.
    Args:
        nursing_notes: List of lists, where each inner list contains tokenized words for a nursing note.
    Returns:
        List of dictionaries where keys are terms and values are their TW weights for each nursing note.
    See pg7 of OG paper for ref
    """
    # Step 1: Calculate Np (total number of nursing notes for the patient)
    Np = len(nursing_notes)

    # Step 2: Calculate tn (term frequencies across all notes)
    all_terms = [term for note in nursing_notes for term in note]
    tn = Counter(all_terms)  # Frequency of each term across the entire corpus
    # Step 3: Compute TW for each note
    tw_weights = []
    for note in nursing_notes:
        note_counter = Counter(note)  # Frequency of each term in the current note
        note_tw = {}
        for term, fm in note_counter.items():
            if fm > 0:
                weight = (1 + np.log2(fm)) * np.log2(Np / tn[term])
                note_tw[term] = weight
            else:
                note_tw[term] = 0
        tw_weights.append(note_tw)
    return tw_weights





print("hello world")
fcrnn = pd.read_csv('../data/filtered_cleaned_raw_nursing_notes_processed.csv')
fcrnn['processed_text'] = fcrnn['processed_text'].apply(eval)
texts_dict = Dictionary(fcrnn['processed_text'])
corpus = [texts_dict.doc2bow(text) for text in fcrnn['processed_text']]
vocab_size = len(texts_dict)
###############

bow_outpath = '../data/bow_matrix.csv'
tw_outpath = '../data/tw_matrix.csv'
# if not os.path.exists(bow_outpath) or not os.path.exists(tw_outpath):
# print(corpus)
# with open(bow_outpath, 'wb') as f:
#     pickle.dump(corpus, f)
# print(f"Results saved to {bow_outpath}")

# Step 1: Compute Term Weighting (TW)
nursing_notes = fcrnn['processed_text'].tolist()
tw_weights_dicts = term_weighting(nursing_notes)

# Step 2: Convert TW into matrix format (same format as BoW for compatibility)
# Create an empty matrix of size (num_documents x vocab_size)
vocab_size = len(texts_dict)
num_documents = len(nursing_notes)
tw_matrix = np.zeros((num_documents, vocab_size))
# Fill the matrix with weights from TW
for doc_idx, tw_weights in enumerate(tw_weights_dicts):
    for term, weight in tw_weights.items():
        if term in texts_dict.token2id:
            term_id = texts_dict.token2id[term]
            tw_matrix[doc_idx, term_id] = weight

print("tw_matrix created")
# print(tw_matrix)
# with open(tw_outpath, 'wb') as f:
#     pickle.dump(tw_outpath, f)
# print(f"Results saved to {tw_outpath}")


# Step 1: NMF on BoW (150 topics)
num_topics_no_sc = 150
nmf_bow_150 = Nmf(
    corpus=corpus,                 # Sparse BoW corpus
    id2word=texts_dict,            # Gensim Dictionary object
    num_topics=num_topics_no_sc,   # Number of topics
    random_state=42,               # Set random seed for reproducibility
)
print("nmf of bow")
# print(nmf_bow_150)
# Compute document-topic matrix
doc_topics = [nmf_bow_150.get_document_topics(doc) for doc in corpus]
X_nmf_bow = np.zeros((len(corpus), num_topics_no_sc))
for doc_idx, topics in enumerate(doc_topics):
    for topic_id, prob in topics:
        X_nmf_bow[doc_idx, topic_id] = prob
print("nmf_bow created")


with open('../data/X_nmf_bow.pkl', 'wb') as f:  # 'wb' for writing in binary mode
    pickle.dump(X_nmf_bow, f)


print(f"X_nmf_bow (150 topics): {X_nmf_bow.shape}")
# Sample and print rows
print_sample_rows(X_nmf_bow, "X_nmf_bow (150 topics)")

corpus_tw = [
    [(word_id, weight) for word_id, weight in enumerate(doc) if weight > 0]
    for doc in tw_matrix
]

print("Number of documents in corpus_tw:", len(corpus_tw))
print("Sample document (first):", corpus_tw[0]) 
print(f"Number of zeros in corpus_tw: {np.sum(corpus_tw == 0)}")
# Small constant to replace zero rows (e.g., epsilon)
epsilon = 1e-10
# Identify all-zero rows
all_zero_rows = np.all(corpus_tw == 0, axis=1)
corpus_tw[all_zero_rows] = epsilon
print(f"Replaced {np.sum(all_zero_rows)} all-zero rows with a small value.")
from gensim import corpora

texts_dict_tw = corpora.Dictionary()
texts_dict_tw.add_documents([[]] * tw_matrix.shape[1])

# Step 2: Train Gensim NMF
num_topics_no_sc = 150
nmf_tw_150 = Nmf(
    corpus=corpus_tw,
    id2word=texts_dict_tw,
    num_topics=num_topics_no_sc,
    random_state=42,
)
print("nmf tw")
# Step 3: Get Document-Topic Matrix
doc_topics_tw = [nmf_tw_150.get_document_topics(doc) for doc in corpus_tw]

X_nmf_tw = np.zeros((len(corpus_tw), num_topics_no_sc))
for doc_idx, topics in enumerate(doc_topics_tw):
    for topic_id, prob in topics:
        X_nmf_tw[doc_idx, topic_id] = prob

print("Document-Topic Matrix for Term-Weighted Matrix")
print("nmf_tw created")





with open('../data/X_nmf_tw.pkl', 'wb') as f:  # 'wb' for writing in binary mode
    pickle.dump(X_nmf_tw, f)

print(f"X_nmf_tw (150 topics): {X_nmf_tw.shape}")
print_sample_rows(X_nmf_tw, "X_nmf_tw (150 topics)")



