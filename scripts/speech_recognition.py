import speech_recognition as sr


class SpeechRec:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def record_and_transcribe(self):
        try:
            text = "Mă numesc Marius și nu vreau să merg acasă! M-am săturat să merg cu trenul! Mă doare capul... Tu nu?"
#             text = """spaCy is an open-source software library for advanced natural language processing,
# written in the programming languages Python and Cython. The library is published under the MIT license
# and its main developers are Matthew Honnibal and Ines Montani, the founders of the software company Explosion."""
            language = "ro-RO"
            return text, language
            # with sr.Microphone() as source:
            #     print("Adjusting for ambient noise... Please wait.")
            #     self.recognizer.adjust_for_ambient_noise(source, duration=1)
            #     print("Ready to record. Please speak.")
            #
            #     # Record the audio
            #     audio_data = self.recognizer.listen(source)
            #     print("Recording complete. Converting speech to text...")
            #
            #     possible_languages = ["ro-RO", "en-US", "fr-FR"]
            #     for language in possible_languages:
            #         try:
            #             text = self.recognizer.recognize_google(audio_data, language=language)
            #             print(f"Recognized the {language} in {text}!")
            #             return text, language
            #         except sr.UnknownValueError:
            #             print(f"Try to match another language")
            #             continue
            #         except sr.RequestError as e:
            #             print(f"Error with language {language}: {e}")
            #             continue
            #     return "There is no language detected from the input audio... Try again!", None

        except sr.UnknownValueError:
            return "Speech Recognition could not understand the audio.", None

        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}", None
