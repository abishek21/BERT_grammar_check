import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np

# Copy the model to the CPU
device = torch.device("cpu")
output_dir="model/"
# Load a trained model and vocabulary that you have fine-tuned
model = BertForSequenceClassification.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)
MAX_LEN=64


def check(sent):
    # Create sentence and label lists
    model.to(device)
    sentences = str(sent)
    ##labels = [1]

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    encoded_sent = tokenizer.encode(sentences, add_special_tokens=True)

    input_ids.append(encoded_sent)
    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN,
                              dtype="long", truncating="post", padding="post")
    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # Convert to tensors.
    prediction_inputs = torch.tensor(input_ids)
    prediction_inputs=prediction_inputs.long()
    prediction_masks = torch.tensor(attention_masks)
    # prediction_labels = torch.tensor(labels)
    # Set the batch size.
    batch_size = 1

    # Create the DataLoader.
    prediction_data = TensorDataset(prediction_inputs, prediction_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    # Prediction on test set

    #print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        # label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        # true_labels.append(label_ids)

    #print('    DONE.')
    # For each input batch...
    for i in range(len(predictions)):
        # The predictions for this batch are a 2-column ndarray (one column for "0"
        # and one column for "1"). Pick the label with the highest value and turn this
        # in to a list of 0s and 1s.
        pred_labels_i = np.argmax(predictions[i], axis=1).flatten()

    ## Combine the predictions for each batch into a single list of 0s and 1s.
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    if flat_predictions[0] == 0:
        p="Grammatically Incorrect sentence"
        return p
    else:
        p="Grammatically Correct sentence"
        return p

string=input("Enter sentence")
answer=check(string)
print(answer)