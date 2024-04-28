import os
from pydub import AudioSegment

# Define the root directory where the IEMOCAP dataset is stored
dataset_root = "/path/to/IEMOCAP"

# Define the session and dialogue directories to extract data from
session_dirs = ["Ses01", "Ses02", "Ses03", "Ses04", "Ses05"]
dialogue_dirs = ["dialog01", "dialog02", "dialog03", "dialog04", "dialog05", "dialog06"]

# Function to extract audio, transcription, and emotion label data
def extract_data():
    for session_dir in session_dirs:
        for dialogue_dir in dialogue_dirs:
            audio_dir = os.path.join(dataset_root, session_dir, dialogue_dir, "sentences/wav")
            transcript_dir = os.path.join(dataset_root, session_dir, "transcriptions")
            emotion_label_dir = os.path.join(dataset_root, session_dir, "EmoEvaluation")

            # Process audio files
            for audio_file in os.listdir(audio_dir):
                if audio_file.endswith(".wav"):
                    audio_path = os.path.join(audio_dir, audio_file)
                    audio = AudioSegment.from_wav(audio_path)
                    # Process the audio as needed (e.g., extract features, analyze)

            # Process transcriptions
            for transcript_file in os.listdir(transcript_dir):
                if transcript_file.endswith(".txt"):
                    transcript_path = os.path.join(transcript_dir, transcript_file)
                    with open(transcript_path, "r") as transcript_file:
                        transcript = transcript_file.read()
                    # Process the transcript as needed

            # Process emotion labels
            for emotion_file in os.listdir(emotion_label_dir):
                if emotion_file.endswith(".txt"):
                    emotion_path = os.path.join(emotion_label_dir, emotion_file)
                    with open(emotion_path, "r") as emotion_file:
                        emotion_data = emotion_file.readlines()
                    # Process the emotion labels as needed

# Run the data extraction function
extract_data()
