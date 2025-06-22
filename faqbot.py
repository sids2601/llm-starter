from transformers import BertTokenizer
from transformers import BertForQuestionAnswering
from transformers import BatchEncoding
import torch

def fetch_tokenizer_for_model(model: str) -> BertTokenizer:
    return BertTokenizer.from_pretrained(model)


def decode_tokens_ids(token_ids, tokenizer: BertTokenizer) -> str:
    return tokenizer.decode(token_ids)


def encode_text(tokenizer: BertTokenizer, question_text: str, model_context: str) -> BatchEncoding:
    return tokenizer.encode_plus(text=question_text, text_pair=model_context)


if __name__ == "__main__":
    model = "bert-large-uncased-whole-word-masking-finetuned-squad"
    bert_model = BertForQuestionAnswering.from_pretrained(model)
    question = "Who founded sunest motors?"
    context = "Sunset Motors is a renowned automobile dealership that has been a cornerstone of the automotive industry since its establishment in 1978. Located in the picturesque town of Crestwood, nestled in the heart of California's scenic Central Valley, Sunset Motors has built a reputation for excellence, reliability, and customer satisfaction over the past four decades. Founded by visionary entrepreneur Robert Anderson, Sunset Motors began as a humble, family-owned business with a small lot of used cars. However, under Anderson's leadership and commitment to quality, it quickly evolved into a thriving dealership offering a wide range of vehicles from various manufacturers. Today, the dealership spans over 10 acres, showcasing a vast inventory of new and pre-owned cars, trucks, SUVs, and luxury vehicles. One of Sunset Motors' standout features is its dedication to sustainability. In 2010, the dealership made a landmark decision to incorporate environmentally friendly practices, including solar panels to power the facility, energy-efficient lighting, and a comprehensive recycling program. This commitment to eco-consciousness has earned Sunset Motors recognition as an industry leader in sustainable automotive retail. Sunset Motors proudly offers a diverse range of vehicles, including popular brands like Ford, Toyota, Honda, Chevrolet, and BMW, catering to a wide spectrum of tastes and preferences. In addition to its outstanding vehicle selection, Sunset Motors offers flexible financing options, allowing customers to secure affordable loans and leases with competitive interest rates."
    tokenizer = fetch_tokenizer_for_model(model)
    encodings = encode_text(tokenizer, question, context)
    tokens = tokenizer.convert_ids_to_tokens(encodings['input_ids'])
    inputs = encodings['input_ids']
    sentence_embeddings = encodings['token_type_ids']
    output = bert_model(input_ids = torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embeddings]))
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    if answer_end >= answer_start:
        answer = ' '.join(tokens[answer_start:answer_end + 1])
    else:
        print("I don't know how to answer this question, can you ask another one?")
    corrected_answer = ''
    for word in answer.split():
        if word[0:2] == '##':
            corrected_answer += word[2:]
        else:
            corrected_answer += ' ' + word
    print(corrected_answer)
