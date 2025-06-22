from transformers import pipeline
sentiment_classifier = pipeline('sentiment-analysis')
ner = pipeline('ner', model='dslim/bert-base-NER')
sequence='One day I will travel the world'
classification=['cooking', 'travel', 'dancing']
zero_shot_classifier = pipeline('zero-shot-classification',model='facebook/bart-large-mnli')
print(zero_shot_classifier(sequence,classification))
