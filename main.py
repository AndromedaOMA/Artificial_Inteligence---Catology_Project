# from scripts import MLNN, PrepareDataSet, SpeechRec
from scripts import SpeechRec, StylometricInfo

if __name__ == "__main__":
    # nn = MLNN()
    # nn.train()
    # p = PrepareDataSet()
    # print(f"Error detections: \n{p.detect_errors_in_data_set()}\n")
    # print(f"No. of instances per class: \n{p.compute_no_of_instances()}\n")
    # print(f"Attributes and their value frequency: \n{p.compute_frequency_of_values()}\n")
    # p.plot_distribution_with_seaborn()
    # p.correct_data_set()
    # p.digit_convertor()
    # p.ohe_plus_one()

    sprec = SpeechRec()
    text, language = sprec.record_and_transcribe()
    if language:
        print(f"Transcribed Text: {text} (Language: {language})")
    else:
        print(text)

    si = StylometricInfo(text, language)
    si.prepare_tokens()
    si.word_length_frequency_plot()
    si.prepare_alpha_tokens()
    si.word_frequency_plot()
    si.alpha_frequency_plot()
    keywords = si.keyword_extraction()
    print(f"Extracted the keywords: {keywords}")
    si.generate_phrase()
    si.nlp_processing()
