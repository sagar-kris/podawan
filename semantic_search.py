import logging
import os
import json
import re
import numpy as np
import pandas as pd
# from sentence_transformers import SentenceTransformer
# from sentence_transformers.util import cos_sim
import tiktoken
import pinecone
import openai

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
# OPENAI_API_KEY = 'for-debugging-purposes'
# get api key from app.pinecone.io
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# find your environment next to the api key in pinecone console
PINECONE_ENV = os.environ.get('PINECONE_ENV')

# embedding model to use
ENCODING_NAME = "cl100k_base"
EMBEDDING_MODEL = "text-embedding-ada-002"
model_max_seq_len = 600
max_token_length = 50000
# EMBEDDING_MODEL = "all-mpnet-base-v2"
# model = SentenceTransformer(EMBEDDING_MODEL)
# model_max_seq_len = 384
# model_dimensions = 768

index = pinecone.Index("youtube-transcripts")

def get_token_count(string_list: list[str], encoding_name: str=ENCODING_NAME) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    string = ''.join(string_list)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_embedding_inner(string: str) -> list[float]:
    response = openai.Embedding.create(
        input=string,
        model=EMBEDDING_MODEL,
    )
    return response['data'][0]['embedding']

def get_embedding(string_list: list[str]) -> list[list[float]]:
    assert get_token_count(string_list) < max_token_length # cap at 50k tokens for now, total cost ~ $0.02
    embeddings = []
    for string in string_list:
        embeddings.append(get_embedding_inner(string))
    return embeddings

# index health
def pinecone_index_health():
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index_str = pinecone.list_indexes()[0]
    index = pinecone.Index(index_str)
    index_stats_response = index.describe_index_stats()
    logging.info(f'pinecone list_indexes(): {str(index_stats_response)}')
    return index_stats_response

# push embeddings to pinecone
def pinecone_upsert(embeddings, namespace: str):
    logging.info(f'number of embeddings: {len(embeddings)}')
    vectors = [(str(i), list(embeddings[i])) for i in range(len(embeddings))]
    index = pinecone.Index(pinecone.list_indexes()[0])
    upsert_response = index.upsert(
        vectors=vectors,
        namespace=namespace
    )
    return upsert_response

# get top k similar chunks as query
def pinecone_query(query_embedding, namespace: str, top_k: int=80):
    index = pinecone.Index(pinecone.list_indexes()[0])
    query_response = index.query(vector=query_embedding,
            top_k=top_k,
            namespace=namespace)
    return query_response
