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

# Configure the NLTK data directory (AI assisted)
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

    #Construção das listas dos documentos, queries e qrels
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
            "qid": docId,
            "query": question
        })

        qrels[docId] = {
            docId: 1
        }


    #Transformações em Dataframes
    corpusDataframe = pd.DataFrame(corpus)
    corpusDataframe["tokens"] = corpusDataframe["text"].apply(preprocess_text)

    queriesDataframe = pd.DataFrame(queries)
    queriesDataframe["tokens"] = queriesDataframe["query"].apply(preprocess_text)

    # print(corpusDataframe)
    # print(queriesDataframe)

    return corpusDataframe, queriesDataframe, qrels


def indexing(corpus, index_path="./index_pubmedqa"):
    documents = []
    
    #Construção do dicionário
    for docno, tokens in zip(corpus["docno"], corpus["tokens"]):
        document = {
            "docno": str(docno),
            "toks": dict(Counter(tokens))
        }

        documents.append(document)

    #Definição do indexer
    indexer = pt.terrier.IterDictIndexer(
        index_path,
        overwrite=True,
        pretokenised=True,
        blocks=False
    )

    #Indexação
    index_ref = indexer.index(documents)
    index = pt.IndexFactory.of(index_ref)

    return index


# Defenição do modelo bm25 (AI assisted)
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

#Praparação para a utilização do modelo e a sua utilização
def run_bm25(queriesDataframe, model):
    #Junção dos tokens das queries
    processed_queries = queriesDataframe["tokens"].apply(lambda tokens: " ".join(tokens))

    #Criação da Dataframe topics
    topics = queriesDataframe[["qid"]].copy()
    topics["query"] = processed_queries

    #Utilização do modelo
    results = model.transform(topics[["qid", "query"]])

    #print(results)

    return results


def evaluation_pytrec(results, qrels):
    #Tranformar a Dataframe num dicionário
    run_pytrec = {}
    for qid, docno, score in zip(results["qid"],results["docno"],results["score"]):
        qid = str(qid)
        docno = str(docno)
        score = float(score)

        if qid not in run_pytrec:
            run_pytrec[qid] = {}

        run_pytrec[qid][docno] = score

    #Avaliação, com o calculo das médias de cada métrica
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        {'ndcg_cut_5', 'ndcg_cut_10', 'map', 'recall_100'}
    )

    evaluation_results = evaluator.evaluate(run_pytrec)

    metrics = {
        'NDCG@5': np.mean([q['ndcg_cut_5'] for q in evaluation_results.values()]),
        'NDCG@10': np.mean([q['ndcg_cut_10'] for q in evaluation_results.values()]),
        'MAP': np.mean([q['map'] for q in evaluation_results.values()]),
        'Recall@100': np.mean([q['recall_100'] for q in evaluation_results.values()]),
    }

    #Cáculo do MRR@10 separado (AI motivated)
    run_top10 = {
        qid: dict(
            sorted(docs.items(), key=lambda item: item[1], reverse=True)[:10]
        )
        for qid, docs in run_pytrec.items()
    }

    mrr_evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'recip_rank'})
    mrr_results = mrr_evaluator.evaluate(run_top10)

    metrics['MRR@10'] = np.mean([q['recip_rank'] for q in mrr_results.values()])

    return metrics


# Execute the pipeline
corpusDataframe, queriesDataframe, qrels = prepare_dataset()

index = indexing(corpusDataframe)

# Create the BM25 model
model = bm25(index, k1=1.2, b=0.75, top_k=100)

# Run retrieval
results = run_bm25(queriesDataframe, model)

metrics = evaluation_pytrec(results, qrels)

print("\n" + "="*50)
print(" TABELA DE RESULTADOS DA MILESTONE 1 (pytrec_eval)")
print("="*50)
for metric, score in metrics.items():
    print(f" {metric:<15}: {score:.4f}")
print("="*50)
