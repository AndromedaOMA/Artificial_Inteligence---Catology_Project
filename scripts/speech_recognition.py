import speech_recognition as sr


class SpeechRec:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def record_and_transcribe(self):
        try:
            with sr.Microphone() as source:
                print("Adjusting for ambient noise... Please wait.")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("Ready to record. Please speak.")

                # Record the audio
                audio_data = self.recognizer.listen(source)
                print("Recording complete. Converting speech to text...")

                # Recognize (convert audio to text)
                text = self.recognizer.recognize_google(audio_data, language="ro-RO")
                return text

        except sr.UnknownValueError:
            return "Speech Recognition could not understand the audio."

        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"
