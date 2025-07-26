import spacy
import pytextrank
from demo import BasicSkillExtractor
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS

class HybridSkillExtractor:
    def __init__(self):
        # Initialize the rule-based keyword extractor
        self.rule_based = BasicSkillExtractor()

        # Load spaCy's English language model
        self.nlp = spacy.load("en_core_web_sm")

        # Create custom stopwords set (from spaCy + domain-specific additions)
        self.custom_stopwords = set(SPACY_STOP_WORDS)
        additional_stopwords = {
            "we", "are", "have", "experience", "looking", "someone", "requirement",
            "preferred", "candidate", "you", "skills", "background", "ability", "knowledge"
        }
        self.custom_stopwords.update(additional_stopwords)

        # Mark additional words as stopwords in spaCy's vocabulary
        for word in additional_stopwords:
            lex = self.nlp.vocab[word]
            lex.is_stop = True

        # Add TextRank component to the pipeline
        self.nlp.add_pipe("textrank")

    def extract_skills(self, text, topn_textrank=15):
        """
        Extract skill-related phrases from job text using both rule-based
        matching and graph-based ranking (TextRank).
        """
        # 1. Use rule-based extractor
        rule_skills = set(self.rule_based.extract_all_skills(text))

        # 2. Use TextRank extractor with stopword filtering
        textrank_skills = set()
        doc = self.nlp(text)

        for phrase in doc._.phrases[:topn_textrank]:
            cleaned_phrase = phrase.text.strip().lower()

            # Filter out very short or very long phrases, or phrases made only of stopwords
            if (
                1 < len(cleaned_phrase) <= 50 and
                len(cleaned_phrase.split()) <= 5 and
                not all(token.text.lower() in self.custom_stopwords for token in self.nlp(cleaned_phrase))
            ):
                textrank_skills.add(cleaned_phrase)

        # 3. Merge rule-based and graph-based results
        combined = sorted(rule_skills.union(textrank_skills))

        return {
            "rule_based_skills": sorted(rule_skills),
            "textrank_phrases": sorted(textrank_skills),
            "combined_skills": combined
        }


