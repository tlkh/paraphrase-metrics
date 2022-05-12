import spacy
import textdistance
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from collections import defaultdict
import statistics
from statistics import stdev
    
rouge = Rouge()
    
def preprocess_text(input_text):
    input_text = str(input_text).strip()
    input_text = input_text.replace("``", "''").replace("‘‘", '"').replace("’’", '"').replace("''", '"')
    input_text = input_text.replace("[", "").replace("]", "")
    input_text = input_text.replace(" .", ".").replace(" ,", ",")
    input_text = input_text.replace("’", "'").replace("“", '"').replace("”", '"')
    return input_text.replace("  ", " ")

def mean_and_stddev(input_list):
    mean = sum(input_list)/len(input_list)
    stddev = stdev(input_list)
    return {"mean": mean, "stddev": stddev}

def rouge_l(s, t):
    return rouge.get_scores(t, s)[0]["rouge-l"]["f"]

def self_bleu(s, t):
    reference = [s.split(" ")]
    candidate = t.split(" ")
    return sentence_bleu(reference, candidate)

def edit_distance(s, t):
    dist_1 = textdistance.damerau_levenshtein.normalized_distance(s, t)
    return dist_1

def relative_position_shift(tokens_i, tokens_j):
    dist_i_list = []
    for p_i in tokens_i:
        dist_ij_list = [abs(p_i-p_j) for p_j in tokens_j]
        dist_i_list.append(min(dist_ij_list))
    delta_W = sum(dist_i_list)/len(dist_i_list)
    return delta_W

def wpd(s, t):
    """
    Word Position Deviation
    Inputs: 2 sentences (SpaCy sequences)
    Output: score (float)
    """
    # collate lemmatized words and normalized positions
    # translate normalized positions to [0.0, 1.0]
    s_token_data = defaultdict(list)
    t_token_data = defaultdict(list)
    s_len, t_len = len(s)-1, len(t)-1
    for i, s_token in enumerate(s):
        s_token_data[s_token.lemma_].append(i/s_len)
    for i, t_token in enumerate(t):
        t_token_data[t_token.lemma_].append(i/t_len)
    delta_W_list = []
    for i in list(s_token_data.keys()):
        # check if word is in both sentences
        if i in list(t_token_data.keys()):
            if len(s_token_data[i])==1 and len(t_token_data[i])==1:
                # simple case (1 instance in each sentence)
                # relative position shift = finding distance
                combined_position_list = s_token_data[i]+t_token_data[i]
                delta_W = max(combined_position_list) - min(combined_position_list)
                delta_W_list.append(delta_W)
            else:
                # multiple occurences - compute relative position shift
                # function is not symmetric:
                # we need to start from sentence with more instances of the word
                if len(s_token_data[i]) >= len(t_token_data[i]):
                    tokens_i = s_token_data[i]
                    tokens_j = t_token_data[i]
                else:
                    tokens_i = t_token_data[i]
                    tokens_j = s_token_data[i]
                delta_W = relative_position_shift(tokens_i, tokens_j)
                delta_W_list.append(delta_W)
    if len(delta_W_list) > 0:
        sigma_W = sum(delta_W_list)/len(delta_W_list)
    else:
        print("Warning: len(delta_W_list)==0")
        print("Since there is no match in structure found, score=1.0")
        sigma_W = 1.0
    return sigma_W

def check_not_punc(text):
    for char in text:
        if char.isalnum():
            return True
    return False

def ld(s, t):
    """
    Lexical Deviation
    Inputs: 2 sentences (SpaCy sequences)
    Output: score (float)
    """
    s_lemma_list = []
    for i, s_token in enumerate(s):
        lemma = s_token.lemma_
        if check_not_punc(lemma):
            s_lemma_list.append(lemma)
    t_lemma_list = []
    for i, t_token in enumerate(t):
        lemma = t_token.lemma_
        if check_not_punc(lemma):
            t_lemma_list.append(lemma)
    total_vocab = list(set(list(s_lemma_list+t_lemma_list)))
    overlap_vocab = []
    for i in s_lemma_list:
        if i in t_lemma_list:
            overlap_vocab.append(i)
    overlap_vocab = list(set(overlap_vocab))
    """print(overlap_vocab)
    non_overlap = []
    for i in total_vocab:
        if i not in overlap_vocab:
            non_overlap.append(i)
    print(non_overlap)
    print(total_vocab)"""
    if len(total_vocab) > 0:
        return 1 - (len(overlap_vocab)/len(total_vocab))
    else:
        print("Warning: overlap_vocab==0")
        print("Since there is no match in vocab found, score=1.0")
        return 1.0