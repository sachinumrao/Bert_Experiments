import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

# Load the model and tokenizer
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Load data for QA
question = """
What is India's expected GDP?
"""

context = """
The International Monetary Fund (IMF) has slashed India's GDP growth projection to 1.9 per cent in 2020 from 5.8 per cent estimated in January, as the global economy hits the worst recession since the Great Depression in the 1930s due to the raging coronavirus pandemic.
Similarly, the World Bank has estimated that India's economy will grow between 1.5 to 2.8 per cent in 2020-21 -- the worst growth performance since the 1991 liberalisation.
"""

# Preprocess the data
input_ids = tokenizer.encode(question, context)
tokens = tokenizer.convert_ids_to_tokens(input_ids)

for token, index in zip(tokens, input_ids):
    print('{:<12} {:>6,}'.format(token, index))

sep_index = input_ids.index(tokenizer.sep_token_id)
num_seg_a = sep_index + 1
num_seg_b = len(input_ids) - num_seg_a
segment_ids = [0]*num_seg_a + [1]*num_seg_b

# Check preprocessing
assert len(input_ids) == len(segment_ids)

# Run model to get start and end index
start_scores, end_scores = model(torch.tensor([input_ids]),
                                 token_type_ids=torch.tensor([segment_ids]))

# Convert index to tokens 
answer_start = torch.argmax(start_scores)
answer_end = torch.argmax(end_scores)

# Join tokens to formulate answer
answer = ''.join(tokens[answer_start:answer_end])

print(answer)

