import numpy as np
import re
import pandas as pd
import nltk
import pyterrier as pt
import pytrec_eval

from pathlib import Path
from collections import Counter
from datasets import load_dataset
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Configure the NLTK data directory
nltk_data_dir = Path(__file__).resolve().parent / ".venv" / "nltk_data"
nltk.data.path.insert(0, str(nltk_data_dir))

stop_words = set(stopwords.words("english"))
stop_words.difference_update({"no", "nor", "not"})

stemmer = PorterStemmer()


# Função para transformar o texto corrido em tokens
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()

    tokens = wordpunct_tokenize(text)

    # Remove pontuação
    tokens = [token for token in tokens if any(char.isalnum() for char in token)]

    # Remove as stop words
    tokens = [token for token in tokens if token not in stop_words]

    # Reduz as palavras ao radical
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens


# Função que prepara o dataset
def prepare_dataset():
    print("Loading the dataset...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    corpus = []
    queries = []
    qrels = {}

    for item in dataset:
        docId = str(item["pubid"])
        context_list = item["context"]["contexts"]
        context_united = " ".join(context_list)
        question = item["question"]

        corpus.append({
            "docno": docId,
            "text": context_united
        })

        queries.append({
            "qId": docId,
            "query": question
        })

        qrels[docId] = {
            docId: 1
        }


    corpusDataframe = pd.DataFrame(corpus)
    corpusDataframe["tokens"] = corpusDataframe["text"].apply(preprocess_text)

    queriesDataframe = pd.DataFrame(queries)
    queriesDataframe["tokens"] = queriesDataframe["query"].apply(preprocess_text)

    print(corpusDataframe)
    print(queriesDataframe)

    return corpusDataframe, queriesDataframe, qrels


def indexing(df_corpus, index_path="./index_pubmedqa"):
    documents = [
        {
            "docno": str(docno),
            "toks": dict(Counter(tokens))
        }
        for docno, tokens in zip(df_corpus["docno"], df_corpus["tokens"])
    ]

    indexer = pt.terrier.IterDictIndexer(
        index_path,
        overwrite=True,
        pretokenised=True,
        blocks=False
    )

    index_ref = indexer.index(documents)
    index = pt.IndexFactory.of(index_ref)

    return index


# Defenição do modelo bm25
def bm25(index, k1=1.2, b=0.75, top_k=100):
    model = pt.terrier.Retriever(
        index,
        wmodel="BM25",
        num_results=top_k,
        controls={
            "bm25.k_1": str(k1),
            "bm25.b": str(b)
        }
    )

    return model


def run_bm25_evaluation(queriesDataframe, model):
    processed_queries = queriesDataframe["tokens"].apply(
        lambda tokens: " ".join(tokens)
    )

    topics = queriesDataframe[["qId"]].copy()
    topics = topics.rename(columns={"qId": "qid"})
    topics["query"] = processed_queries

    results = model.transform(topics[["qid", "query"]])
    print(results)

    return results

def evaluation_pytrec(results, qrels):
    run_pytrec = {}
    for _, row in results.iterrows():
        qid = str(row['qid'])
        docno = str(row['docno'])
        score = float(row['score'])
        
        if qid not in run_pytrec:
            run_pytrec[qid] = {}
        run_pytrec[qid][docno] = score
    
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        {'ndcg_cut_5', 'ndcg_cut_10', 'map', 'recall_100'}
    )

    evaluation_results = evaluator.evaluate(run_pytrec)

    # Aggregate the mean of all queries
    metrics = {
        'NDCG@5': np.mean([q['ndcg_cut_5'] for q in evaluation_results.values()]),
        'NDCG@10': np.mean([q['ndcg_cut_10'] for q in evaluation_results.values()]),
        'MAP': np.mean([q['map'] for q in evaluation_results.values()]),
        'Recall@100': np.mean([q['recall_100'] for q in evaluation_results.values()]),
    }



# Execute the pipeline
corpusDataframe, queriesDataframe, qrels = prepare_dataset()

index = indexing(corpusDataframe)

# Create the BM25 model
model = bm25(index, k1=1.2, b=0.75, top_k=100)

# Run retrieval
results = run_bm25_evaluation(queriesDataframe, model)