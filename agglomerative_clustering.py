import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from models import InferSent
import torch
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc


phrases = [
'what encloses something?',
'what does something enclose?',

'what controls something?',
'what does something control?',
'where does something control something?',

'what Controls something?',
'what does something Control?',



'who monitors something?',
'what does someone monitor?',
'who controls something?',
'what does someone control?',
'where does someone control something?',
'what mays?',
'what does something may?',
'what does something may to do?',
'What coordinates something?',
'What does something coordinate?',
'what requires something?',
'what does something require?',
'what provides something?',
'what does something provide?',
'where does something provide something?'
'what synthesizes something?',
'what does something synthesize?',
'how does something synthesize something?'
'what does something synthesize something from?',
'what uses something?',
'what is using something?',
'how is something being used?',
'what is being used?',
'what moves something?',
'where is something being moved?',
'what is being moved?',
'where does something move something?',
'what does something move something through?']


prefix_phrases = ["i want to " + p for p in phrases]


#######################################################################################################################
# Model = SentenceBERT:
model_SentenceBERT = SentenceTransformer('bert-base-nli-mean-tokens')

# Model = Universal Sentence Encoder:
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model_Universal = hub.load(module_url)
print("module %s loaded" % module_url)
#######################################################################################################################


def df_model_embed(model, sentences_index, sentences_input):
    """
    This function takes phrases/sentences and create Dataframe with their embeddings

    :param model: The sentence embedding model used (SentenceBERT/InferSent/Universal Sentence Encoder)
    :param sentences_index: The sentences which will be used as Indices for the Dataframe (since with the prefix is too long).
    :param sentences_input: The phrases/sentences the model generates embedding for.
    :return: Dataframe containing the sentences themselves and their embeddings.
    """
    if model is model_Universal:
        embed = model(sentences_input)
    else:
        embed = model.encode(sentences_input)

    df = pd.DataFrame(data=embed, index=sentences_index)
    if sentences_input is prefix_phrases:
        phrases_type = "with added prefix: 'I want to'"
    else:
        phrases_type = ""

    return df, phrases_type


def plot_dendrogram(df, phrases_type, model_type, method, metric):
    """
    This function plots the dendrogram.
    :param df: Dataframe containing the sentces embeddings
    :param phrases_type: determine if the list of sentences contain the prefix "I want to" or not (for title needs)
    :param model_type: the name of the model (for title needs)
    :param method: the method used for the calculation of the linkage matrix.
    :param metric: the metric used to calculate the distances between the embeddings.
    :return: plot of dendrogram with relevant information
    """
    plt.figure(figsize=(10, 7))
    title = "Dendrograms of Sentences Embeddings\n" + phrases_type + "\nmodel: " + model_type + ", Linkage's Method: " + method +\
            " ,Metric: " + metric
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    # Z = shc.linkage(df, method='average', metric='cosine')
    Z = shc.linkage(df, method=method, metric=metric)
    dend = shc.dendrogram(Z, labels=pd.Index.tolist(df.index), leaf_rotation=45,
                          leaf_font_size=7)
    plt.show()


# def agg_cluster(df):
#     # model = AgglomerativeClustering(linkage='single', distance_threshold=0, n_clusters=None, affinity='cosine')
#     model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
#     clustering = model.fit(df)
#     return clustering


def main():
    df_Universal_pre, phrases_type_Universal_pre = df_model_embed(model_Universal, phrases, prefix_phrases)
    plot_dendrogram(df_Universal_pre, phrases_type_Universal_pre, "Universal Sentence Encoder", 'single', 'cosine')
    print()
    df_bert, phrases_type_bert = df_model_embed(model_SentenceBERT, phrases, phrases)
    df_bert_pre, phrases_type_bert_pre = df_model_embed(model_SentenceBERT, phrases, prefix_phrases)

    df_Universal, phrases_type_Universal = df_model_embed(model_Universal, phrases, phrases)
    df_Universal_pre, phrases_type_Universal_pre = df_model_embed(model_Universal, phrases, prefix_phrases)

    plot_dendrogram(df_Universal, phrases_type_Universal, "Universal Sentence Encoder", 'average', 'cosine')
    plot_dendrogram(df_Universal_pre, phrases_type_Universal_pre, "Universal Sentence Encoder", 'average', 'cosine')

    plot_dendrogram(df_bert, phrases_type_bert, "SentenceBERT", 'average', 'cosine')
    plot_dendrogram(df_bert_pre, phrases_type_bert_pre, "SentenceBERT", 'average', 'cosine')
    print()


    if __name__ == "__main__":
        main()
