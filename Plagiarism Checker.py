import torch
from transformers import BertTokenizer, BertModel
def preprocess_text(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer.tokenize(text)
    max_length = 128
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        tokens += ['[PAD]'] * (max_length - len(tokens))
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(input_ids).unsqueeze(0) 
    return input_ids
def calculate_cosine_similarity(text1, text2):
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    input_ids1 = preprocess_text(text1)
    input_ids2 = preprocess_text(text2)
    with torch.no_grad():
        outputs1 = model(input_ids1)
        outputs2 = model(input_ids2)
    embeddings1 = outputs1.last_hidden_state.mean(dim=1)
    embeddings2 = outputs2.last_hidden_state.mean(dim=1)
    similarity = torch.cosine_similarity(embeddings1, embeddings2).item()
    return similarity
def is_plagiarized(text1, text2, threshold=0.9):
    similarity = calculate_cosine_similarity(text1, text2)
    if similarity >= threshold:
        return True
    else:
        return False
if __name__ == "__main__":
    text1 = "Friends are the Family you Choose."
    text2 = "True Friends Are Always Together In Spirit."
    text3 = "True Friends Are Never Apart,Maybe In Distance But Never In Heart."
    print("Text 1:", text1)
    print("Text 2:", text2)
    print("Text 3:", text3)
    print("Similarity between text 1 and text 2:", calculate_cosine_similarity(text1, text2))
    print("Similarity between text 1 and text 3:", calculate_cosine_similarity(text1, text3))
    print("Is text 1 plagiarized from text 2?", is_plagiarized(text1, text2))
    print("Is text 1 plagiarized from text 3?", is_plagiarized(text1, text3))
