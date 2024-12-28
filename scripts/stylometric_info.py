from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt
from collections import Counter
from rake_nltk import Rake
# nltk.download('punkt_tab')
nltk.download('stopwords')


class StylometricInfo:
    def __init__(self, text, language):
        self.alpha_tokens_frequency = None
        self.filtered_tokens = None
        self.text = text
        self.language = language

    def prepare_tokens(self):
        clean_text = self.text
        lang = self.language

        # tokens = word_tokenize(clean_text, language=lang)
        tokens = word_tokenize(clean_text)

        # no punctuation
        self.filtered_tokens = ([token for token in tokens if token.isalpha()])
        print(f"Prepared the tokens: {self.filtered_tokens}")

    def prepare_alpha_tokens(self):
        clean_tokens = self.filtered_tokens
        alpha_dict = {}

        for token in clean_tokens:
            for c in token:
                alpha_dict[c.lower()] = alpha_dict.get(c.lower(), 0) + 1

        print(f"Prepared the frewquency of chars: {alpha_dict}")

        self.alpha_tokens_frequency = alpha_dict.values()

    def word_length_frequency_plot(self):
        token_lengths = [len(token) for token in self.filtered_tokens]
        length_frequency_distribution = nltk.FreqDist(token_lengths)
        length_frequency_distribution.plot(15, title="Frequency of tokens length distribution")
        plt.show()

    def word_frequency_plot(self):
        length_frequency_distribution = nltk.FreqDist(self.filtered_tokens)
        length_frequency_distribution.plot(15, title="Frequency of tokens distribution")
        plt.show()

    def alpha_frequency_plot(self):
        length_frequency_distribution = nltk.FreqDist(self.alpha_tokens_frequency)
        length_frequency_distribution.plot(15, title="Frequency of characters distribution")
        plt.show()

    def keyword_extraction(self):
        match self.language:
            case "ro-RO": stopwords = [
                "un", "o", "aceasta", "acesta", "care", "despre", "este", "sunt", "in", "pe", "din", "si", "de", "la",
                "cu", "noi", "al", "ai", "este", "pentru", "mai", "nu", "cat", "culoare", "cu", "mane", "blana", "lunga", "scurta",
                "mare", "mica", "gri", "alb", "negru", "rosu", "tip", "colorat", "si", "vital", "animale", "domestice", "specie", "rasă",
                "ochi", "urechi", "bot", "picioare", "coada", "permanent", "activ"]
            case "fr-FR": stopwords = [
                "un", "une", "des", "le", "la", "les", "cette", "ce", "il", "elle", "nous", "vous", "ils", "elles", "de", "dans",
                "pour", "sur", "avec", "par", "comme", "est", "sont", "a", "à", "au", "aux", "être", "avoir", "lui", "leur", "ses",
                "les", "cette", "autre", "chat", "chatte", "grande", "petit", "moyenne", "longue", "courte", "poils", "noir", "blanc",
                "gris", "rouge", "caractère", "type", "pattes", "nez", "yeux", "intelligent", "calme", "actif", "mignon"]
            case "en-US": stopwords = [
                "a", "an", "the", "and", "or", "but", "with", "in", "on", "at", "of", "for", "to", "by", "is", "are", "was",
                "were", "am", "i", "you", "he", "she", "we", "they", "my", "his", "her", "its", "their", "that", "this", "which",
                "has", "have", "had", "it", "these", "those", "be", "being", "been", "has", "have", "having", "cat", "breed", "color",
                "long", "short", "fur", "eyes", "ears", "tail", "coat", "paws", "gray", "black", "white", "brown", "striped",
                "tabby", "fluffy", "active", "calm", "friendly", "playful", "smart", "beautiful", "pet", "animal", "domestic"]
        r = Rake(stopwords=stopwords)
        r.extract_keywords_from_text(self.text)
        return r.get_ranked_phrases()

    def generate_phrase(self):
        pass
