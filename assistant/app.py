# app.py

import gradio as gr
import noisereduce as nr
import tempfile
from gtts import gTTS
from nlp_logic import get_answer_basic
from asr import load_and_preprocess_audio, transcribe_cleaned_audio, clean_transcription

def process_voice_command(audio_file_path):
    if audio_file_path is None:
        return "Ndagusabye uvuge.", None

    user_text = ""
    response_audio_path = None
    assistant_response_text = ""
    # bite
    # imana
    # neza
    try:
        waveform, sr = load_and_preprocess_audio(audio_file_path)
        user_text = transcribe_cleaned_audio(waveform, sr)
        if not user_text:
            user_text = "[Ntacyo wigeze uvuga]"
            assistant_response_text = "Ndagusabye uvuge neza."
    except Exception as e:
        print(f"ASR error: {e}")
        user_text = "[Ikosa ryo kumva]"
        assistant_response_text = "Habayeho ikibazo kumva ibyo wavuze."

    if user_text and user_text not in ["[Ntacyo wigeze uvuga]", "[Ikosa ryo kumva]"]:
        assistant_response_text = get_answer_basic(user_text)

    if assistant_response_text:
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                response_audio_path = fp.name
                tts = gTTS(text=assistant_response_text)
                tts.save(response_audio_path)
        except Exception as e:
            print(f"TTS error: {e}")
            response_audio_path = None
    print("audio path:", response_audio_path)
    return clean_transcription(user_text), response_audio_path

audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Vuga hano")
text_output = gr.Textbox(label="Ibyo wavuze")
audio_output = gr.Audio(label="Ibisubizo bya Assistant")

iface = gr.Interface(
    fn=process_voice_command,
    inputs=audio_input,
    outputs=[text_output, audio_output],
    title="🎙️ Umufasha wo kuvuga mu Kinyarwanda 🤖",
    description="Vuga ikibazo cyawe mu Kinyarwanda. Umufasha azagusubiza akoresheje QA isanzwe."
)

if __name__ == "__main__":
    iface.launch(inline=False)
