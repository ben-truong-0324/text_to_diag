#NMF with SC
from gensim.test.utils import datapath
from gensim.models import NMF
from gensim.corpora.dictionary import Dictionary
import pandas as pd
from gensim.test.utils import datapath
from gensim.matutils import sparse2full


import pickle
import numpy as np
import os
import pandas as pd

# with open(f'{os.getcwd()}/doc2vec_dataset_full.pkl', 'rb') as f:
#   df_X = pickle.load(f)
print("hello world")
fcrnn = pd.read_csv('../data/filtered_cleaned_raw_nursing_notes_processed.csv')
fcrnn['processed_text'] = fcrnn['processed_text'].apply(eval)
texts_dict = Dictionary(fcrnn['processed_text'])
corpus = [texts_dict.doc2bow(text) for text in fcrnn['processed_text']]
vocab_size = len(texts_dict)

bow_matrix = np.array([sparse2full(doc, vocab_size) for doc in corpus])
print("bow matrix created")
print(bow_matrix)
bow_outpath = '../data/bow_matrix.csv'
with open(bow_outpath, 'wb') as f:
    pickle.dump(bow_matrix, f)
print(f"Results saved to {bow_outpath}")

from collections import Counter

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

# Step 1: Compute Term Weighting (TW)
nursing_notes = fcrnn['processed_text'].tolist()
tw_weights_dicts = term_weighting(nursing_notes)

# Step 2: Convert TW into matrix format (same format as BoW for compatibility)
# Create an empty matrix of size (num_documents x vocab_size)
vocab_size = len(texts_dict)
num_documents = len(nursing_notes)

# Initialize a zero matrix
tw_matrix = np.zeros((num_documents, vocab_size))

# Fill the matrix with weights from TW
for doc_idx, tw_weights in enumerate(tw_weights_dicts):
    for term, weight in tw_weights.items():
        if term in texts_dict.token2id:
            term_id = texts_dict.token2id[term]
            tw_matrix[doc_idx, term_id] = weight

print("tw_matrix created")
print(tw_matrix)
tw_outpath = '../data/tw_matrix.csv'
with open(tw_outpath, 'wb') as f:
    pickle.dump(tw_outpath, f)
print(f"Results saved to {tw_outpath}")
# Define number of topics for the two configurations
num_topics_no_sc = 150

# Step 1: NMF on BoW (150 topics)
nmf_bow_150 = NMF(n_components=num_topics_no_sc, random_state=42)
X_nmf_bow = nmf_bow_150.fit_transform(bow_matrix)  # Document-topic matrix for BoW (150 topics)

# Step 2: NMF on TW (150 topics)
nmf_tw_150 = NMF(n_components=num_topics_no_sc, random_state=42)
X_nmf_tw = nmf_tw_150.fit_transform(tw_matrix)  # Document-topic matrix for TW (150 topics)


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


# Final Output
print(f"X_nmf_bow (150 topics): {X_nmf_bow.shape}")
print(f"X_nmf_tw (150 topics): {X_nmf_tw.shape}")

# Sample and print rows
print_sample_rows(X_nmf_bow, "X_nmf_bow (150 topics)")
print_sample_rows(X_nmf_tw, "X_nmf_tw (150 topics)")


with open('X_nmf_bow.pkl', 'wb') as f:  # 'wb' for writing in binary mode
    pickle.dump(X_nmf_bow, f)
with open('X_nmf_tw.pkl', 'wb') as f:  # 'wb' for writing in binary mode
    pickle.dump(X_nmf_tw, f)
