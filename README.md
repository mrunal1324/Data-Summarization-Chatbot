
# Data Summarization Chatbot

This project is a user-friendly chatbot capable of summarizing various types of data, including textual data, audio files, psychological profiles, and map data. The chatbot uses powerful AI models to process and generate concise summaries, making complex information easier to understand.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Detailed Options](#detailed-options)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)

## Features

• Text Summarization: Summarizes long texts into concise and clear summaries.

• Audio Transcription and Summarization: Transcribes spoken audio and provides a summary.

• Psychological Profile Summarization: Generates a summary for provided psychological profile data.

• Map Data Summarization: Provides summaries for locations and routes from map data.



## Getting Started

By following the steps below, you can set it up on your local machine and start summarizing data.
## Installation

To install and run this chatbot, follow the steps below:

1. Prerequisites
Ensure you have Python installed on your system (version 3.7 or later). You can download Python here.

  2. Clone the Repository
Clone this project from GitHub using the following command:
bash
Copy code

git clone https://github.com/mrunal1324/Data-Summarization-Chatbot

  3. Install Required Libraries
Navigate to the project folder and install the necessary dependencies using:
bash
Copy code
pip install -r requirements.txt

  4. Download Additional Models
The chatbot uses pre-trained models for summarization and audio transcription. These models are automatically downloaded when the script is first run.


## Usage/Examples

To start the chatbot, navigate to the project directory and run the script:
bash
Copy code

python chatbot.py

The chatbot interface will open in your terminal, providing you with several options to choose from.

Detailed Options
Once the chatbot starts, you can choose from the following options:
1.	Text Summarization
Enter a long piece of text, and the chatbot will provide a concise summary.

2.	Audio Transcription and Summarization
Enter the path to an audio file (in mp3 format), and the chatbot will transcribe and summarize the spoken content.

3.	Psychological Profile Summarization
Enter psychological profile data (e.g., notes, observations), and the chatbot will summarize it effectively.

4.	Map Data Summarization
Provide map data in JSON format, and the chatbot will generate summaries for locations and routes.

5.	Exit
Choose this option to exit the chatbot.

Input Formats

• Audio: Provide the file path to an mp3 file.

• Text: Simply type or paste your text into the prompt.

• Map Data: Provide JSON formatted map data like the example below:

Copy code

{
  "locations": [
    { "name": "Shaniwar Wada", "type": "Historical Fort", "coordinates": "18.5196, 73.8553" },
    { "name": "Aga Khan Palace", "type": "Palace", "coordinates": "18.5523, 73.8991" }
  ],
  "routes": [
    { "start": "Shaniwar Wada", "end": "Aga Khan Palace", "distance_km": 8.5 }
  ]
}

## Dependencies
This project uses the following libraries and models:

•	transformers: For text summarization using models like facebook/bart-large-cnn and google/flan-t5-large.

•	whisper: For audio transcription using OpenAI’s Whisper model.

•	json: For handling JSON data input.

You can find all these dependencies listed in the requirements.txt file.

## Troubleshooting

1. Model Download Issues:
If the models do not download automatically, ensure you have a stable internet connection and sufficient disk space.

2. Audio File Not Recognized:
Ensure your audio file is in mp3 format and the file path is correct.

3. Invalid JSON Format for Map Data:
Double-check the JSON structure to ensure it is correctly formatted. You can validate JSON here.

4. Other Errors:
If you encounter other errors, please check the error message for guidance or consult the project documentation.
