# Podcast Padawan

Get a summarized explanation of any YouTube podcast episode.
Get answers to your questions for any topics discussed in the episode.

<img width="851" alt="Screen Shot 2023-08-03 at 12 20 39 AM" src="https://github.com/sagar-kris/podawan/assets/9098686/5b62b01a-281d-4e53-81ef-0de8a6203e45">

## Architecture

![Screen Shot 2023-07-29 at 8 40 00 PM](https://github.com/sagar-kris/podawan/assets/9098686/9adff299-0a1f-40bf-b5ca-691656a00fc8)

## Setup

To install the required packages for this plugin, run the following command:

```bash
pip install -r requirements.txt
```

To run the plugin, enter the following command:

```bash
python main.py
```

Once the local server is running:

1. Navigate to https://chat.openai.com. 
2. In the Model drop down, select "Plugins" (note, if you don't see it there, you don't have access yet).
3. Select "Plugin store"
4. Select "Develop your own plugin"
5. Enter in `localhost:5003` since this is the URL the server is running on locally, then select "Find manifest file".

The plugin should now be installed and enabled!

Sample query: "what advice did nims have for joe in https://www.youtube.com/watch?v=xqtRV7jWOzk"

## Getting help

Raise an issue in this repo
