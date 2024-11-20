from gensim.test.utils import datapath
from gensim.models import Nmf
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import datapath
from gensim.matutils import sparse2full
from gensim import corpora

import pandas as pd
import pickle
import numpy as np
import os
import pandas as pd
from collections import Counter

target_label_code_groups = {
    1: [1, 139, "Infectious and Parasitic Diseases"],
    2: [140, 239, "Neoplasms"],
    3: [240, 279, "Endocrine, Nutritional, Metabolic, Immunity"],
    4: [280, 289, "Blood and Blood-Forming Organs"],
    5: [290, 319, "Mental Disorders"],
    6: [320, 389, "Nervous System and Sense Organs"],
    7: [390, 459, "Circulatory System"],
    8: [460, 519, "Respiratory System"],
    9: [520, 579, "Digestive System"],
    10: [580, 629, "Genitourinary System"],
    11: [630, 677, "Pregnancy, Childbirth, and the Puerperium"],
    12: [680, 709, "Skin and Subcutaneous Tissue"],
    13: [710, 739, "Musculoskeletal System and Connective Tissue"],
    14: [740, 759, "Congenital Anomalies"],
    15: [780, 789, "Symptoms"],
    16: [790, 796, "Nonspecific Abnormal Findings"],
    17: [797, 799, "Ill-defined and Unknown Causes of Morbidity and Mortality"],
    18: [800, 999, "Injury and Poisoning"],
    # 19: [0, 0, "Reference or Supplemental V-Codes"]
}

def get_multigroup_label(icd_code):
    """
    return None for: cannot get first 3 numbers (null value) OR number does not match any groups (invalid code)
    """
    # if letter, return group 19 as they start with letter
    if isinstance(icd_code, str) and icd_code[0].isalpha():
        return 19
    try:
        code_num = int(str(icd_code)[:3])  # Extract first 3 characters as an integer
    except ValueError:
        return None  # Return None if conversion fails
    # Iterate through each group in target_label_code_groups to find the correct range
    for label, (start, end, description) in target_label_code_groups.items():
        if isinstance(start, int) and code_num in range(start, end + 1):  # +1 to include the end value
            return label
    return None  

def calculate_visit_columns(group):
    hadm_ids = sorted(group['HADM_ID'].unique())
    first_hadm = hadm_ids[0]
    # All non-first HADM_IDs, excluding the first (lowest) one
    all_non_first_hadm = hadm_ids[1:] if len(hadm_ids) > 1 else hadm_ids
    # Add new columns to each row in the group
    group['FIRST_VISIT_HADM_ID'] = first_hadm
    return group

def calculate_diag_groups(group):
    # Unique values of MULTIGROUP_LABEL_TARGET for the first_hadm_id
    first_hadm_id = group['FIRST_VISIT_HADM_ID'].iloc[0]
    diag_groups_first_hadm = group[group['HADM_ID'] == first_hadm_id]['MULTIGROUP_LABEL_TARGET'].unique().tolist()
    return pd.DataFrame({
        'SUBJECT_ID': [group['SUBJECT_ID'].iloc[0]],
        'DIAG_GROUPS_OF_FIRST_HADM_ONLY': [diag_groups_first_hadm],
    })

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

print("reading diagnoses and fcrnn")
diag_df = pd.read_csv('../data/DIAGNOSES_ICD.csv')
diag_df['MULTIGROUP_LABEL_TARGET'] = diag_df['ICD9_CODE'].apply(get_multigroup_label).fillna(-1).astype(int)
diag_df = diag_df.groupby('SUBJECT_ID', group_keys=False).apply(calculate_visit_columns)
multi_group_label_df = diag_df.groupby('SUBJECT_ID', group_keys=False).apply(calculate_diag_groups).reset_index(drop=True)

fcrnn = pd.read_csv('../data/filtered_cleaned_raw_nursing_notes_processed.csv')
fcrnn['processed_text'] = fcrnn['processed_text'].apply(eval)

X_nmf_bow_path = '../data/X_nmf_bow.pkl'
nmf_bow_dataset_full_path = '../data/nmf_bow_dataset.pkl'
nmf_tw_dataset_full_path = '../data/nmf_tw_dataset.pkl'
bow_outpath = '../data/bow_matrix.csv'
tw_outpath = '../data/tw_matrix.csv'

############### create bow
texts_dict = Dictionary(fcrnn['processed_text'])
corpus = [texts_dict.doc2bow(text) for text in fcrnn['processed_text']]
vocab_size = len(texts_dict)
############### calc tw
nursing_notes = fcrnn['processed_text'].tolist()
tw_weights_dicts = term_weighting(nursing_notes)
num_documents = len(nursing_notes)
tw_matrix = np.zeros((num_documents, vocab_size))
# Fill the matrix with weights from TW
for doc_idx, tw_weights in enumerate(tw_weights_dicts):
    for term, weight in tw_weights.items():
        if term in texts_dict.token2id:
            term_id = texts_dict.token2id[term]
            tw_matrix[doc_idx, term_id] = weight
print("tw_matrix created")

# Step 1: NMF on BoW (150 topics)
num_topics_no_sc = 150
if not os.path.exists(X_nmf_bow_path):
    print("working on X_nmf_bow")
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
    with open(X_nmf_bow_path, 'wb') as f:  # 'wb' for writing in binary mode
        pickle.dump(X_nmf_bow, f)
    print(f"X_nmf_bow (150 topics): {X_nmf_bow.shape}")
    print_sample_rows(X_nmf_bow, "X_nmf_bow (150 topics)")
try:
    if not os.path.exists(nmf_bow_dataset_full_path):
        print("working on nmf_bow_dataset_full")
        with open(X_nmf_bow_path, 'rb') as f:
            X_nmf_bow = pickle.load(f)
        X_nmf_bow = pd.DataFrame(X_nmf_bow)
        if isinstance(X_nmf_bow, pd.DataFrame):
            X_nmf_bow['SUBJECT_ID'] = fcrnn['SUBJECT_ID']
            # Merge with multi_group_label_df on SUBJECT_ID
            nmf_bow_dataset_full = pd.merge(X_nmf_bow, multi_group_label_df, on='SUBJECT_ID', how='inner')
            # Check the result of the merge
            print(nmf_bow_dataset_full.shape)
            print(nmf_bow_dataset_full.head())
            with open(nmf_bow_dataset_full_path, 'wb') as f:  # 'wb' for writing in binary mode
                pickle.dump(nmf_bow_dataset_full, f)
        else:
            print("X_nmf_bow unable to convert to DataFrame")
except Exception as e:
    print(e)

print("working on corpus_tw")

corpus_tw = [
    [(word_id, weight) for word_id, weight in enumerate(doc)]  # Replace with actual weights
    for doc in tw_matrix  # Assuming tw_matrix is your matrix with weights
]

print("Number of documents in corpus_tw:", len(corpus_tw))
epsilon = 1e-10
corpus_tw = [
    doc if len(doc) > 0 and any(weight > 0 for _, weight in doc) else [(0, epsilon)]
    for doc in corpus_tw
]
print(f"Replaced all-zero rows with a small value.")
from gensim.models import TfidfModel
tfidf_model = TfidfModel(corpus_tw)  # If you want TF-IDF weighting
corpus_tw = tfidf_model[corpus_tw]  # Apply TF-IDF to the corpus

texts_dict_tw = corpora.Dictionary(corpus_tw)  # Create a dictionary for TW

# Step 2: Train Gensim NMF
print("training gensim nmf")
num_topics_no_sc = 150
nmf_tw_150 = Nmf(
    corpus=corpus_tw,
    id2word=texts_dict_tw,
    num_topics=num_topics_no_sc,
    random_state=42,
)
print("nmf tw done")
# Step 3: Get Document-Topic Matrix
doc_topics_tw = [nmf_tw_150.get_document_topics(doc) for doc in corpus_tw]
X_nmf_tw = np.zeros((len(corpus_tw), num_topics_no_sc))
for doc_idx, topics in enumerate(doc_topics_tw):
    for topic_id, prob in topics:
        X_nmf_tw[doc_idx, topic_id] = prob

print("Document-Topic Matrix for Term-Weighted Matrix")
print("X_nmf_tw created")
X_nmf_tw_path = '../data/X_nmf_tw.pkl'
with open(X_nmf_tw_path, 'wb') as f:  # 'wb' for writing in binary mode
    pickle.dump(X_nmf_tw, f)

print(f"X_nmf_tw (150 topics): {X_nmf_tw.shape}")
print_sample_rows(X_nmf_tw, "X_nmf_tw (150 topics)")
try:
    if not os.path.exists(nmf_tw_dataset_full_path):
        print("working on nmf_tw_dataset_full")
        with open(X_nmf_tw_path, 'rb') as f:
            X_nmf_tw = pickle.load(f)
        X_nmf_tw = pd.DataFrame(X_nmf_tw)
        if isinstance(X_nmf_tw, pd.DataFrame):
            X_nmf_tw['SUBJECT_ID'] = fcrnn['SUBJECT_ID']
            # Merge with multi_group_label_df on SUBJECT_ID
            nmf_tw_dataset_full = pd.merge(X_nmf_tw, multi_group_label_df, on='SUBJECT_ID', how='inner')
            # Check the result of the merge
            print(nmf_tw_dataset_full.shape)
            print(nmf_tw_dataset_full.head())
            with open(nmf_tw_dataset_full_path, 'wb') as f:  # 'wb' for writing in binary mode
                pickle.dump(nmf_tw_dataset_full, f)
        else:
            print("X_nmf_tw unable to convert to DataFrame")
except Exception as e:
    print(e)

##############################################
# with SC
def compute_pmi_for_terms(terms, term_matrix):
    """
    Compute Pointwise Mutual Information (PMI) for a pair of terms based on their joint frequency in the term-document matrix.
    """
    term_counts = term_matrix.sum(axis=0)  # Frequency of each term
    total_count = term_matrix.sum()  # Total number of terms in all documents
    pmi_matrix = np.zeros((len(terms), len(terms)))
    pr_ij_matrix = np.zeros((len(terms), len(terms)))
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            term_i = terms[i]
            term_j = terms[j]
            pr_ij = (term_matrix[:, term_i] * term_matrix[:, term_j]).sum() / total_count
            # Marginal probabilities for terms i and j
            pr_i = term_counts[term_i] / total_count
            pr_j = term_counts[term_j] / total_count
            pr_ij_matrix[i, j] = pr_ij
            pr_ij_matrix[j, i] = pr_ij
            # Calculate PMI
            if pr_ij > 0:
                pmi = np.log2(pr_ij / (pr_i * pr_j))
                pmi_matrix[i, j] = pmi
                pmi_matrix[j, i] = pmi  # Symmetric matrix
    return pmi_matrix, pr_ij_matrix



def compute_sc_for_topic(topic_idx, num_top_terms, term_matrix):
    """
    Compute the Semantic Coherence (SC) for a given topic based on its top-n terms.
    Args:
        topic_idx: The index of the topic.
        num_top_terms: Number of top terms to consider for SC.
        term_matrix: Term-document matrix (BoW or TW matrix).
    Returns:
        sc: The semantic coherence for the given topic.
    """
    # Get the indices of the top-n terms for the topic
    topic_terms = np.argsort(term_matrix[topic_idx, :])[::-1][:num_top_terms]
    # Compute the PMI matrix for these top terms
    pmi_matrix,  = compute_pmi_for_terms(topic_terms, term_matrix)
    npmi_values = []
    for i in range(len(topic_terms)):
        for j in range(i + 1, len(topic_terms)):
            pmi = pmi_matrix[i, j]
            pr_ij = pr_ij_matrix[i,j]
            # Calculate NPMI (Normalized PMI)
            if pmi != 0:
                npmi = pmi / (-np.log2(pr_ij)) if pmi_matrix[i, j] > 0 else 0
                npmi_values.append(npmi)

    # SC is the sum of all NPMI values divided by the number of choices
    num_pairs = len(npmi_values)
    num_pairs = len(topic_terms) * (len(topic_terms) - 1) // 2 #n choose 2 is n(n-1)/2
    sc = np.sum(npmi_values) / num_pairs if num_pairs > 0 else 0
    return sc

def compute_semantic_coherence_for_all_topics(nmf_model, term_matrix, num_topics, num_top_terms):
    """
    Compute Semantic Coherence (SC) for all topics in an NMF model.
    Args:
        nmf_model: The fitted NMF model.
        term_matrix: The BoW or TW matrix.
        num_topics: Number of topics.
        num_top_terms: Number of top terms per topic to use for SC computation.
    Returns:
        sc_scores: List of SC scores for each topic.
    """
    sc_scores = []
    for topic_idx in range(num_topics):
        sc = compute_sc_for_topic(topic_idx, num_top_terms, term_matrix, term_matrix.shape[1])
        sc_scores.append(sc)
    return sc_scores

num_topics = 100
num_top_terms = 10  ##################DIS TRU THO?

# Compute SC for all topics
sc_bow_scores = compute_semantic_coherence_for_all_topics(nmf_bow, bow_matrix, num_topics, num_top_terms)
sc_tw_scores = compute_semantic_coherence_for_all_topics(nmf_tw, tw_matrix, num_topics, num_top_terms)

def adjust_matrix_using_sc(matrix, sc_scores):
    """
    Adjust the matrix using the SC scores for each topic.
    Args:
        matrix: The BoW or TW matrix.
        sc_scores: List of SC scores for each topic.
    Returns:
        adjusted_matrix: Adjusted matrix incorporating SC.
    """
    # Create a diagonal matrix with the SC scores
    sc_matrix = np.diag(sc_scores)
    # Adjust the original matrix by multiplying it with the SC matrix ##############################dis tru tho?
    adjusted_matrix = matrix @ sc_matrix
    return adjusted_matrix

# Adjust BoW and TW matrices using SC scores
adjusted_bow_matrix = adjust_matrix_using_sc(bow_matrix, sc_bow_scores)
adjusted_tw_matrix = adjust_matrix_using_sc(tw_matrix, sc_tw_scores)


# Apply NMF
nmf_bow_sc = NMF(n_components=100, random_state=42)
X_nmf_bow_sc = nmf_bow_sc.fit_transform(adjusted_bow_matrix)

nmf_tw_sc = NMF(n_components=100, random_state=42)
X_nmf_tw_sc = nmf_tw_sc.fit_transform(adjusted_tw_matrix)


# Output the shape of the resulting matrices
print(f"X_nmf_bow_sc (100 topics): {X_nmf_bow_sc.shape}")
print(f"X_nmf_tw_sc (100 topics): {X_nmf_tw_sc.shape}")