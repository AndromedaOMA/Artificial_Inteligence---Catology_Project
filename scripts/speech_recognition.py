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

                possible_languages = ["ro-RO", "en-US", "fr-FR"]
                for language in possible_languages:
                    try:
                        text = self.recognizer.recognize_google(audio_data, language=language)
                        print(f"Recognized the {language} in {text}!")
                        return text, language
                    except sr.UnknownValueError:
                        print(f"Try to match another language")
                        continue
                    except sr.RequestError as e:
                        print(f"Error with language {language}: {e}")
                        continue
                return "There is no language detected from the input audio... Try again!", None

        except sr.UnknownValueError:
            return "Speech Recognition could not understand the audio.", None

        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}", None
