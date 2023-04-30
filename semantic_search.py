# import pinecone
import os
import json
import re
# import numpy as np
# import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

PINECONE_API_KEY = "4e2d748f-61c3-4fa3-94fb-6b454ef6c8a3"
PINECONE_ENVIRONMENT = "northamerica-northeast1-gcp"

# get api key from app.pinecone.io
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY') or 'PINECONE_API_KEY'
# find your environment next to the api key in pinecone console
PINECONE_ENV = os.environ.get('PINECONE_ENVIRONMENT') or 'PINECONE_ENVIRONMENT'

model = SentenceTransformer('all-mpnet-base-v2')
model_max_seq_len = 384
model_dimensions = 768
# index = pinecone.Index("youtube-transcripts")

# # index health
# def pinecone_index_health():
#   pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
#   pinecone.list_indexes()
#   index_stats_response = index.describe_index_stats()
#   return index_stats_response
