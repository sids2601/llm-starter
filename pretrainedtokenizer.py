import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModelForSequenceClassification

def fetch_tokenizer_for_model(model: str)-> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model)

def fetch_tokens(sentence: str, tokenizer: PreTrainedTokenizerBase)-> list[str]:
    return tokenizer.tokenize(sentence)

def decode_tokens_ids(token_ids, tokenizer: PreTrainedTokenizerBase)->str:
    return tokenizer.decode(token_ids)

if __name__ == "__main__":
    model = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = fetch_tokenizer_for_model(model)
    sentence = "I am excited to learn AI"
    input_ids_pt = tokenizer(sentence, return_tensors="pt")
    sequence_model = AutoModelForSequenceClassification.from_pretrained(model)
    with torch.no_grad(): # Disabling gradient tracking since we are using inference and not training. Saves memory and speeds up computation
        logits = sequence_model(**input_ids_pt).logits #Unpacking dictionary

    predicted_class_id = logits.argmax().item() # Finds the index of the highest score
    print(sequence_model.config.id2label[predicted_class_id])



