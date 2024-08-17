from transformers import pipeline
import whisper
import json
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
psych_summarizer = pipeline("summarization", model="google/flan-t5-large")

# Initialize Whisper model for transcription
whisper_model = whisper.load_model("medium")

# Helper function to summarize text
def summarize_text(text, max_length=130, min_length=30):
    return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

# Helper function to transcribe and summarize audio
def transcribe_and_summarize_audio(audio_file):
    result = whisper_model.transcribe(audio_file)
    transcribed_text = result['text'].strip()
    if len(transcribed_text.split()) > 5:
        input_length = len(transcribed_text.split())
        max_length = max(20, int(min(0.5 * input_length, 100)))
        min_length = max(5, int(0.3 * max_length))
        summary = summarize_text(transcribed_text, max_length, min_length)
        return transcribed_text, summary
    else:
        return transcribed_text, "Text is too short to summarize meaningfully."

# Helper function to summarize psychological profile
def summarize_psych_profile(profile_text):
    return psych_summarizer(profile_text, max_length=50, min_length=20, repetition_penalty=2.0, do_sample=True, top_k=50)[0]['summary_text']

# Helper function to summarize map data
def summarize_map_data(map_data):
    locations = map_data["locations"]
    routes = map_data["routes"]

    location_summary = "\n".join([f"- {loc['name']} ({loc['type']}) at coordinates {loc['coordinates']}" for loc in locations])
    route_summary = "\n".join([f"- From {route['start']} to {route['end']}, distance: {route['distance_km']} km" for route in routes])

    summary_text = f"""
    Map Data Summary:
    Locations:
    {location_summary}

    Routes:
    {route_summary}
    """
    return summarize_text(summary_text, max_length=100, min_length=50)

# Main function to run the code as a script
def main():
    while True:
        print("\nChoose an option:")
        print("1. Text Summarization")
        print("2. Audio Transcription and Summarization")
        print("3. Psychological Profile Summarization")
        print("4. Map Data Summarization")
        print("5. Exit")

        choice = input("\nEnter your choice: ")

        if choice == '1':
            text_input = input("\nEnter text for summarization:\n")
            if text_input:
                summary = summarize_text(text_input)
                print("\nSummary:")
                print(summary)
            else:
                print("Please enter some text.")

        elif choice == '2':
            audio_file_path = input("\nEnter the path to the audio file (mp3 format): ")
            try:
                if audio_file_path:
                    transcribed_text, summary = transcribe_and_summarize_audio(audio_file_path)
                    print("\nTranscribed Text:")
                    print(transcribed_text)
                    print("\nSummary:")
                    print(summary)
                else:
                    print("Please provide a valid audio file path.")
            except Exception as e:
                print(f"An error occurred: {e}")

        elif choice == '3':
            profile_input = input("\nEnter psychological profile data:\n")
            if profile_input:
                summary = summarize_psych_profile(profile_input)
                print("\nSummary:")
                print(summary)
            else:
                print("Please enter psychological profile data.")

        elif choice == '4':
            map_data_input = input("\nEnter map data in JSON format (e.g., {'locations': [...], 'routes': [...]})\n")
            try:
                map_data = json.loads(map_data_input)
                summary = summarize_map_data(map_data)
                print("\nSummary:")
                print(summary)
            except json.JSONDecodeError:
                print("Invalid JSON format. Please check your input.")
            except Exception as e:
                print(f"An error occurred: {e}")

        elif choice == '5':
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
