import logging
import json
import re
import numpy as np
import pandas as pd

import quart
import quart_cors
from quart import request
import uvicorn
import gunicorn
# from mangum import Mangum

from youtube_transcript_api import YouTubeTranscriptApi

# from semantic_search import model, model_max_seq_len, index, pinecone_index_health
from semantic_search import (model_max_seq_len, get_embedding, get_embedding_inner,
                             pinecone_index_health, pinecone_upsert, pinecone_query)

app = quart_cors.cors(quart.Quart(__name__), allow_origin="https://chat.openai.com")

@app.get("/podcasts")
async def get_podcast():
    podcast = request.args.get('podcast')
    prompt = request.args.get('prompt')
    return quart.Response(response=json.dumps({"response": getPodcastData(podcast, prompt)}), status=200)

@app.get("/logo.png")
async def plugin_logo():
    filename = 'logo.png'
    return await quart.send_file(filename, mimetype='image/png')

@app.get("/.well-known/ai-plugin.json")
async def plugin_manifest():
    host = request.headers['Host']
    with open("./.well-known/ai-plugin.json") as f:
        text = f.read()
        return quart.Response(text, mimetype="text/json")

@app.get("/openapi.yaml")
async def openapi_spec():
    host = request.headers['Host']
    with open("openapi.yaml") as f:
        text = f.read()
        return quart.Response(text, mimetype="text/yaml")
    
def getPodcastTranscript(podcast_id: str) -> list[str]:
    transcript_json = YouTubeTranscriptApi.get_transcript(podcast_id, languages=['en', 'en-US', 'es'])
    
    # return the text in the transcript
    transcript = []
    for item in transcript_json:
       transcript.append(str(item["text"]))
    
    # consolidate into single string
    transcript = ' '.join(transcript)
    
    # clean up newline chars
    transcript = re.sub("\n", " ", transcript)
    transcript = re.sub("\'", "", transcript)
    
    # split on period punctuation
    transcript_dup = transcript.split(".")
    transcript = []
    for sentence in transcript_dup:
       transcript.append(f'{sentence}.')
    
    # combine sentences into pairs
    si = iter(transcript)
    transcript = [sentence + next(si, '') for sentence in si]
    
    # chunk long sentences to stay within model limits
    transcript_final = []
    for item in transcript:
        if len(item) > model_max_seq_len: # break it up
            # print(f"len of item is {len(item)}")
            num_chunks = int(len(item)/model_max_seq_len) + 1
            # print(f"num chunks is {num_chunks}")
            offset = int(model_max_seq_len - (((model_max_seq_len * num_chunks) - len(item))/(num_chunks - 1))) + 1
            # print(f"offset is {offset}")
            for idx in range(0, num_chunks):
                # print(f"inner loop count is {count}")
                starting_idx = idx * offset
                chunk = item[starting_idx : starting_idx + model_max_seq_len]
                transcript_final.append(chunk)
        else:
            transcript_final.append(item)

    return transcript_final

def getPodcastData(podcast: str, prompt: str) -> list[str]:
    podcast_id = podcast.split("=")[1]
    namespaces = list(pinecone_index_health()['namespaces'].keys())

    sentences = getPodcastTranscript(podcast_id)

    # create and upsert index
    if podcast_id not in namespaces:
        logging.info(f'creating new namespace')
        embeddings = get_embedding(sentences)

        logging.info(f'batching and upserting to pinecone')
        num_batches = int(len(embeddings)/250) + 1
        for i in range(num_batches):
            embeddings_batch = embeddings[i*250:(i+1)*250]
            upsert_response = pinecone_upsert(embeddings_batch, podcast_id)
            logging.info(f'pinecone upsert response: {upsert_response}')

    # retrieve top k similar chunks from pinecone index
    query_embedding = get_embedding_inner(prompt)
    vectors_top_k = pinecone_query(query_embedding, podcast_id)['matches']
    # sort by chronological order using id
    vectors_top_k.sort(key=lambda x: int(x['id']))
    logging.info(f'query response sorted: {vectors_top_k}')
    # get text strings from vector IDs
    sentences_top_k = [sentences[int(vector['id'])] for vector in vectors_top_k]
    sentences_top_k = ''.join(sentences_top_k)
    logging.info(f'query response final: {sentences_top_k}')

    return sentences_top_k

    # sim = np.zeros(len(sentences))

    # for i in range(len(sentences)):
    #     sim[i] = cos_sim(query_embedding, embeddings[i])

    # sorted_inds = np.argsort(sim)[::-1][:80] # take the 100 most relevant embeddings
    # sorted_inds = np.sort(sorted_inds)
    # logging.info(f'number of sorted inds: {len(sorted_inds)}')

    # sentences_final = []
    # for ind in sorted_inds:
    #     sentences_final.append(sentences[ind])

# handler = Mangum(app)

def main():
    # pinecone_index_health()
    # app.run(debug=True, host="0.0.0.0", port=5003)
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("main:app", port=5003, log_level="debug")

if __name__ == "__main__":
    main()
