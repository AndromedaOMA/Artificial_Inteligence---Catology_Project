# from scripts import MLNN, PrepareDataSet, SpeechRec
from scripts import SpeechRec

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
    result = sprec.record_and_transcribe()
    print("Transcription:", result)
