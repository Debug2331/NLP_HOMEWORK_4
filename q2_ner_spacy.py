"""
Part II, Q2: Named Entity Recognition with spaCy.
Setup:  pip install spacy
        python -m spacy download en_core_web_sm
"""
import spacy

# Load the small English pipeline (includes the NER component)
nlp = spacy.load("en_core_web_sm")

text = (" LeBron James as the NBA player of the United States in Laker "
        "and won the NBA Final MVP in 2020.")

doc = nlp(text)

print(f"{'Entity':<22}{'Label':<10}{'Start':<8}{'End'}")
for ent in doc.ents:
    # ent.text = entity string, ent.label_ = entity type,
    # start_char / end_char = character offsets in the original string
    print(f"{ent.text:<22}{ent.label_:<10}{ent.start_char:<8}{ent.end_char}")
