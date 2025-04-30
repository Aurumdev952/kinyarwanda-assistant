# Mini Kinyarwanda Voice Assistant

This project implements a basic voice assistant pipeline for the Kinyarwanda language, simulating the core components of an intelligent robot's voice interaction capabilities: Automatic Speech Recognition (ASR), Natural Language Processing (NLP) for understanding, and Text-to-Speech (TTS) for responding.

It was developed as an assignment for the Intelligent Robotics course (Assigned: April 23, 2025, Due: April 30, 2025).

## Features

*   **ASR:** Converts Kinyarwanda speech to text using a fine-tuned Whisper model (`benax-rw/KinyaWhisper`).
*   **NLP:** Understands simple Kinyarwanda questions by matching transcriptions against a predefined dictionary of Q&A pairs.
*   **TTS:** Generates spoken Kinyarwanda responses using the Coqui TTS library.
*   **UI:** Provides a simple web interface using Gradio to interact with the voice assistant via microphone

## Technologies Used

*   Python 3
*   Hugging Face `transformers` (for KinyaWhisper ASR)
*   `pytorch` (backend for transformers)
*   `gTTS` (for Kinyarwanda TTS)
*   `gradio` (for the web UI)
*   `librosa`, `soundfile`, `torchaudio` (for audio handling)
*   `conda` (for environment management)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Aurumdev952/kinyarwanda-assistant.git assistant
    cd assistant
    ```

2.  **Create and activate the Conda environment:**
    Ensure you have Conda installed.
    ```bash
    conda env create -f transformers-audio.yml
    conda activate transformers-audio
    ```

3.  **Verify Dependencies:** The `transformers-audio.yml` file should handle most dependencies. If you encounter issues, ensure `pytorch` (CPU version is specified) and the other listed libraries are installed correctly within the `transformers-audio` environment.

## Running the Application

1.  Activate the conda environment:
    ```bash
    conda activate transformers-audio
    ```
2.  Run the Gradio application script:
    ```bash
    python app.py
    ```
    This will start a local web server, and the Gradio interface URL will be printed in the console (usually `http://127.0.0.1:7860/`). Open this URL in your web browser.

## How to Use

1.  Open the Gradio interface in your browser.
2.  Use the "Record" button to speak your question in Kinyarwanda using your microphone.
3.  The application will process the audio, display the transcription and the chosen answer in the text box, and play the spoken response.

## Data

*   **Kinyarwanda Q&A:** The basic knowledge base is stored in `src/qa_data.py`. It contains a dictionary of predefined Kinyarwanda questions and their corresponding answers. You can expand this dictionary to add more capabilities to the assistant.
*   **Sample Audio:** The `audio_samples/` directory contains sample Kinyarwanda audio files. `audio_samples/audio_transcriptions.txt` lists these files and their human-verified transcriptions.

## Acknowledgments

*   Benax for the KinyaWhisper model fine-tuned on Kinyarwanda.
*   The Hugging Face team for the `transformers` library.
*   The developers of Gradio, Coqui TTS, librosa, etc.

