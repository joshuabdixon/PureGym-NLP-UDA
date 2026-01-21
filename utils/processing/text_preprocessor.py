from __future__ import annotations

from typing import Iterable, Optional, Set, Union

import regex as re
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet 
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)


class TextPreprocessor:
    """
    Text normalisation for downstream modelling.

    Steps
    -----
    1. Normalise common mojibake and remove invisible formatting characters.
    2. Lowercase text, normalise apostrophes and dashes.
    3. Expand contractions and remove possessive markers.
    4. Remove punctuation (pattern-driven) and digits.
    5. Tokenise and lemmatise:
       - If `use_pos_tagging=True`: NLTK POS tagging + WordNet lemmatisation (POS-aware).
       - Else: WordNet lemmatisation (no POS tagging) plus a small auxiliary-verb map.
    6. Remove stopwords (NLTK + `extra_stopwords`).

    Parameters
    ----------
    punctuation_pattern
        Regex pattern (using the `regex` module) for punctuation to remove.
    extra_stopwords
        Extra stopwords to remove (lowercased).
    use_pos_tagging
        If True, use NLTK POS tagging + WordNet lemmatisation (POS-aware).
        If False, use WordNet lemmatisation (no POS tagging).
    language
        NLTK stopword language.

    """

    _AUX_VERB_LEMMA_MAP = {
        "am": "be",
        "is": "be",
        "are": "be",
        "was": "be",
        "were": "be",
        "been": "be",
        "being": "be",
        "has": "have",
        "have": "have",
        "had": "have",
        "having": "have",
        "does": "do",
        "do": "do",
        "did": "do",
        "doing": "do",
    }

    def __init__(
        self,
        punctuation_pattern: str,
        extra_stopwords: Optional[Iterable[str]] = None,
        use_pos_tagging: bool = False,
        language: str = "english",
    ) -> None:
        self._use_pos_tagging = use_pos_tagging

        # Compiled regex
        self._punct_re = re.compile(punctuation_pattern)
        self._numbers_re = re.compile(r"\d+")
        self._format_chars_re = re.compile(r"[\p{Cf}]+")
        self._edge_punct_re = re.compile(r"^\p{P}+|\p{P}+$")

        # Dash normalisation
        self._dash_translate = str.maketrans({"–": "-", "—": "-", "−": "-", "‒": "-"})

        # Contractions/possessives WITH apostrophes (e.g. "can't", "you're", "John's", "parents'")
        self._possessive_re = re.compile(r"\b(\p{L}+)[’']s\b", flags=re.IGNORECASE)
        self._plural_possessive_re = re.compile(r"\b(\p{L}+s)[’']\b", flags=re.IGNORECASE)

        # Irregular negatives handled explicitly
        self._cant_re = re.compile(r"\bcan[’']t\b", flags=re.IGNORECASE)
        self._wont_re = re.compile(r"\bwon[’']t\b", flags=re.IGNORECASE)
        self._shant_re = re.compile(r"\bshan[’']t\b", flags=re.IGNORECASE)

        # General n't forms (e.g. "haven't" -> "have not")
        self._nt_re = re.compile(r"\b(\p{L}+?)n[’']t\b", flags=re.IGNORECASE)

        # Other apostrophe contractions
        self._am_re = re.compile(r"\b(\p{L}+)[’']m\b", flags=re.IGNORECASE)
        self._contractions = [
            (re.compile(r"\b(\p{L}+)[’']re\b", flags=re.IGNORECASE), r"\1 are"),
            (re.compile(r"\b(\p{L}+)[’']ve\b", flags=re.IGNORECASE), r"\1 have"),
            (re.compile(r"\b(\p{L}+)[’']ll\b", flags=re.IGNORECASE), r"\1 will"),
            (re.compile(r"\b(\p{L}+)[’']d\b", flags=re.IGNORECASE), r"\1 would"),
        ]

        # Common contractions WITHOUT apostrophes (e.g. "havent", "cant", "youre", "im")
        self._no_apostrophe_map = {
            "dont": "do not",
            "cant": "can not",
            "wont": "will not",
            "havent": "have not",
            "hasnt": "has not",
            "hadnt": "had not",
            "isnt": "is not",
            "arent": "are not",
            "wasnt": "was not",
            "werent": "were not",
            "doesnt": "does not",
            "didnt": "did not",
            "shouldnt": "should not",
            "wouldnt": "would not",
            "couldnt": "could not",
            "mustnt": "must not",
            "mightnt": "might not",
            "im": "i am",
            "ive": "i have",
            "id": "i would",
            "ill": "i will",
            "youre": "you are",
            "were": "we are",
            "theyre": "they are",
        }

        # Stopwords + lemmatiser
        self._stopwords: Set[str] = set(stopwords.words(language))
        if extra_stopwords:
            self._stopwords |= {str(w).lower().strip() for w in extra_stopwords}

        self._lemmatiser = WordNetLemmatizer()

    @staticmethod
    def _tag_to_wordnet_pos(tag: str) -> str:
        # Map Penn Treebank tags (from nltk.pos_tag) to WordNet POS constants
        if tag.startswith("J"):
            return wordnet.ADJ
        if tag.startswith("V"):
            return wordnet.VERB
        if tag.startswith("N"):
            return wordnet.NOUN
        if tag.startswith("R"):
            return wordnet.ADV
        return wordnet.NOUN

    def _normalise_no_apostrophe_forms(self, text: str) -> str:
        s = text
        for src, tgt in self._no_apostrophe_map.items():
            s = re.sub(rf"\b{re.escape(src)}\b", tgt, s, flags=re.IGNORECASE)
        return s

    def _normalise_contractions(self, text: str) -> str:
        s = text
        s = self._cant_re.sub("can not", s)
        s = self._wont_re.sub("will not", s)
        s = self._shant_re.sub("shall not", s)
        s = self._nt_re.sub(r"\1 not", s)
        s = self._am_re.sub(r"\1 am", s)

        s = self._possessive_re.sub(r"\1", s)
        s = self._plural_possessive_re.sub(r"\1", s)

        for pattern, repl in self._contractions:
            s = pattern.sub(repl, s)

        return s

    def transform_tokens(self, text: Union[str, None]) -> list[str]:
        """
        Transform raw text into cleaned tokens.

        Parameters
        ----------
        text
            Raw text.

        Returns
        -------
        list[str]
            Cleaned tokens.
        """
        if text is None:
            return []

        # Normalise encoding artefacts and invisible characters
        s = str(text).replace("â€™", "’")
        s = self._format_chars_re.sub("", s)

        # Lowercase and normalise apostrophes/dashes
        s = s.lower().replace("’", "'")
        s = s.translate(self._dash_translate)

        # Expand contractions and remove possessives
        s = self._normalise_no_apostrophe_forms(s)
        s = self._normalise_contractions(s)

        # Remove punctuation and digits
        s = self._punct_re.sub(" ", s)
        s = self._numbers_re.sub("", s)

        # Tokenise and lemmatise
        raw_tokens = s.split()

        if self._use_pos_tagging and raw_tokens:
            tagged = nltk.pos_tag(raw_tokens)
            tokens = []
            for tok, tag in tagged:
                wn_pos = self._tag_to_wordnet_pos(tag)
                lemma = self._lemmatiser.lemmatize(tok, pos=wn_pos)
                lemma = self._AUX_VERB_LEMMA_MAP.get(tok, lemma)
                tokens.append(lemma)
        else:
            tokens = [
                self._AUX_VERB_LEMMA_MAP.get(t, self._lemmatiser.lemmatize(t))
                for t in raw_tokens
            ]

        # Stopword removal
        tokens = [
            t for t in tokens
            if self._edge_punct_re.sub("", t) not in self._stopwords
        ]

        return tokens

    def transform(self, text: Union[str, None]) -> str:
        """Transform raw text into a cleaned whitespace-separated string."""
        return " ".join(self.transform_tokens(text))

    def transform_many(self, texts: Iterable[Union[str, None]]) -> list[str]:
        """Transform many texts into cleaned strings."""
        return [self.transform(t) for t in texts]