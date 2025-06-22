from transformers import BertForQuestionAnswering, BatchEncoding
from transformers import BertTokenizer
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def fetch_tokenizer_for_model(model: str) -> BertTokenizer:
    return BertTokenizer.from_pretrained(model)


def decode_tokens_ids(token_ids, tokenizer: BertTokenizer) -> str:
    return tokenizer.decode(token_ids)


def encode_text(tokenizer: BertTokenizer, question_text: str, answer_text: str) -> BatchEncoding:
    return tokenizer.encode_plus(text=question_text, text_pair=answer_text)


if __name__ == "__main__":
    model = "bert-large-uncased-whole-word-masking-finetuned-squad"
    bert_model = BertForQuestionAnswering.from_pretrained(model)
    tokenizer = fetch_tokenizer_for_model(model)
    question = "When was the first DVD released?"
    answer = "The first DVD (Digital Versatile Disc) was released on March 24, 1997. It was a movie titled 'Twister' and was released in Japan. DVDs quickly gained popularity as a replacement for VHS tapes and became a common format for storing and distributing digital video and data."
    encoding = encode_text(tokenizer, question, answer)
    inputs = encoding['input_ids']
    sentence_embeddings = encoding['token_type_ids']
    tokens = tokenizer.convert_ids_to_tokens(inputs)
    output = bert_model(input_ids = torch.tensor([inputs]), token_type_ids = torch.tensor([sentence_embeddings]))
    start_index = torch.argmax(output.start_logits)
    end_index = torch.argmax(output.end_logits)
    s_scores = output.start_logits.detach().numpy().flatten()
    e_scores = output.end_logits.detach().numpy().flatten()
    token_labels = []
    for (i,token) in enumerate(tokens):
        token_labels.append('{:}-{:>2}'.format(token,i))

    ax = sns.barplot(x=token_labels, y=s_scores)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
    ax.grid(True)
    plt.show()

