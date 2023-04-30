import logging
import json
import re
import numpy as np
# import pandas as pd

import quart
import quart_cors
from quart import request
import uvicorn
from mangum import Mangum

from youtube_transcript_api import YouTubeTranscriptApi

from semantic_search import model, model_max_seq_len, index
from sentence_transformers.util import cos_sim

logging.basicConfig(level=logging.DEBUG)

app = quart_cors.cors(quart.Quart(__name__), allow_origin="https://chat.openai.com")

@app.get("/podcasts")
async def get_podcast():
    podcast = request.args.get('podcast')
    prompt = request.args.get('prompt')
    logging.debug(f'arg podcast: {podcast}')
    logging.debug(f'arg prompt: {prompt}')
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
    
def getPodcastTranscript(podcast: str):
    podcast_id = podcast.split("=")[1]
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

def getPodcastData(podcast: str, prompt: str):
    sentences = getPodcastTranscript(podcast)
    embeddings = model.encode(sentences)

    # query_sentence = "what were the key insights for dating women"
    query_embedding = model.encode(prompt)

    sim = np.zeros(len(sentences))

    for i in range(len(sentences)):
        sim[i] = cos_sim(query_embedding, embeddings[i])

    sorted_inds = np.argsort(sim)[::-1][:80] # take the 100 most relevant embeddings
    sorted_inds = np.sort(sorted_inds)
    logging.debug(f'number of sorted inds: {len(sorted_inds)}')

    sentences_final = []
    for ind in sorted_inds:
        sentences_final.append(sentences[ind])

    return sentences_final

handler = Mangum(app)

# def main():
#     # pinecone_index_health()
#     # app.run(debug=True, host="0.0.0.0", port=5003)
#     uvicorn.run("main:app", port=5000, log_level="info")

# if __name__ == "__main__":
#     main()
