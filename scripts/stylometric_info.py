import datetime
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt
# from collections import Counter
from rake_nltk import Rake
from groq import Groq
import stanza
from spacy_stanza import load_pipeline
import rowordnet
import xml.etree.ElementTree as ET

# nltk.download('punkt_tab')
nltk.download('stopwords')
stanza.download('ro')


def print_separator():
    print("-" * 40)


def search_word(word, root):
    results = []
    for synset in root.findall("SYNSET"):
        lemmas = [literal.text for literal in synset.findall("SYNONYM/LITERAL")]
        if word in lemmas:
            gloss = synset.find("DEF").text if synset.find("DEF") is not None else "No definition"
            results.append((synset.get("id"), lemmas, gloss))
    return results


# Extract only synonyms (excluding the searched word)
def get_synonyms(word, root):
    synonyms = set()
    results = search_word(word, root)
    for synset_id, lemmas, gloss in results:
        for lemma in lemmas:
            synonyms.add(lemma)
    return list(synonyms)


class StylometricInfo:
    def __init__(self, text, language):
        self.keywords = None
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
        print_separator()

    def prepare_alpha_tokens(self):
        clean_tokens = self.filtered_tokens
        alpha_dict = {}

        for token in clean_tokens:
            for c in token:
                alpha_dict[c.lower()] = alpha_dict.get(c.lower(), 0) + 1

        print(f"Prepared the frewquency of chars: {alpha_dict}")
        print_separator()
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
            case "ro-RO":
                stopwords = [
                    "un", "o", "aceasta", "acesta", "care", "despre", "este", "sunt", "in", "pe", "din", "si", "de",
                    "la",
                    "cu", "noi", "al", "ai", "este", "pentru", "mai", "nu", "cat", "culoare", "cu", "mane", "blana",
                    "lunga", "scurta",
                    "mare", "mica", "gri", "alb", "negru", "rosu", "tip", "colorat", "si", "vital", "animale",
                    "domestice", "specie", "rasă",
                    "ochi", "urechi", "bot", "picioare", "coada", "permanent", "activ"]
            case "fr-FR":
                stopwords = [
                    "un", "une", "des", "le", "la", "les", "cette", "ce", "il", "elle", "nous", "vous", "ils", "elles",
                    "de", "dans",
                    "pour", "sur", "avec", "par", "comme", "est", "sont", "a", "à", "au", "aux", "être", "avoir", "lui",
                    "leur", "ses",
                    "les", "cette", "autre", "chat", "chatte", "grande", "petit", "moyenne", "longue", "courte",
                    "poils", "noir", "blanc",
                    "gris", "rouge", "caractère", "type", "pattes", "nez", "yeux", "intelligent", "calme", "actif",
                    "mignon"]
            case "en-US":
                stopwords = [
                    "a", "an", "the", "and", "or", "but", "with", "in", "on", "at", "of", "for", "to", "by", "is",
                    "are", "was",
                    "were", "am", "i", "you", "he", "she", "we", "they", "my", "his", "her", "its", "their", "that",
                    "this", "which",
                    "has", "have", "had", "it", "these", "those", "be", "being", "been", "has", "have", "having", "cat",
                    "breed", "color",
                    "long", "short", "fur", "eyes", "ears", "tail", "coat", "paws", "gray", "black", "white", "brown",
                    "striped",
                    "tabby", "fluffy", "active", "calm", "friendly", "playful", "smart", "beautiful", "pet", "animal",
                    "domestic"]
        r = Rake(stopwords=stopwords)
        r.extract_keywords_from_text(self.text)
        self.keywords = [kw for kw in r.get_ranked_phrases() if len(kw.split()) <= 2]
        print(f"Extracted the keywords: {self.keywords}")
        print_separator()

    def generate_phrase(self):
        # keywords_values = [str(keyword) for keyword in self.keywords]
        # print(f"TEST:"
        #       f"text -> {self.text}"
        #       f"kewwords -> {keywords_values}")
        # print(f"Voi furniza o listă de cuvinte cheie: {str(self.keywords)} "
        #       f"și propoziția de unde provin aceste cuvinte cheie "
        #       f"pentru a extrage contextul lor: {str(self.text)}."
        #       f"Generează câte o propoziție pentru fiecare cuvânt cheie, "
        #       f"dar fiecare cuvânt cheie trebuie să-și păstreze contextul din "
        #       f"propoziția inițială. Nu folosi niciun tool suplimentar. "
        #       f"Răspunsurile trebuie să fie doar text simplu.")
        client = Groq(
            api_key="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",
        )
        completion = client.chat.completions.create(
            model="llama3-groq-70b-8192-tool-use-preview",
            messages=[
                {
                    "role": "user",
                    "content": f"Voi furniza o listă de cuvinte cheie: {str(self.keywords)} "
                               f"și propoziția de unde provin aceste cuvinte cheie "
                               f"pentru a extrage contextul lor: {str(self.text)}."
                               f"Generează câte o propoziție pentru fiecare cuvânt cheie, "
                               f"dar fiecare cuvânt cheie trebuie să-și păstreze contextul din "
                               f"propoziția inițială. Nu folosi niciun tool suplimentar. "
                               f"Răspunsurile trebuie să fie doar text simplu."
                }
            ],
            temperature=0.5,
            max_tokens=1024,
            top_p=0.65,
            stream=False,
            stop=None,
        )

        if not completion.choices or not completion.choices[0].message.content:
            raise ValueError("No valid content received from the model.")

        print(completion.choices[0].message.content)
        print_separator()

    def nlp_processing(self):
        # snlp = stanza.Pipeline(lang="ro")
        nlp = load_pipeline("ro")

        doc = nlp(self.text)
        for token in doc:
            print(f"Token: {token.text}, Lemma: {token.lemma_}, POS: {token.pos_}")
        print_separator()

        self.synonym_detector()

    # def synonym_detector(self):
    #     """source: https://github.com/dumitrescustefan/RoWordNet/blob/master/jupyter/synonym_antonym.ipynb"""
    #     wordnet = rowordnet.RoWordNet()
    #     for token in self.filtered_tokens:
    #         synset_id = wordnet.synsets(literal=token)
    #         if synset_id:
    #             print(f"Token: {token}")
    #             synset = wordnet(synset_id[0])
    #             # wordnet.print_synset(synset_id[0])
    #             literals = list(synset.literals)
    #             print(f"Synonym: {str(literals)}")
    #             print_separator()

    def synonym_detector(self):
        tree = ET.parse("RoWordNet/xml/rown.xml")
        root = tree.getroot()
        for token in self.filtered_tokens:
            print(f"Token: {token} -> Synonyms: {get_synonyms(token, root)}")
        print_separator()
