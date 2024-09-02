from deep.punctuationmodel import punctuationmodel
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("oliverguhr/spelling-correction-multilingual-base")
model = AutoModelForSeq2SeqLM.from_pretrained("oliverguhr/spelling-correction-multilingual-base")

def restore_punctuation(input_file, output_file):
    # Load the PunctuationModel
    model = punctuationmodel()

    # Initialize an empty list to hold punctuated text chunks
    punctuated_text_chunks = []

    # Define the maximum chunk size (in characters)
    max_chunk_size = 1024  # Adjust this value based on the model's limitations

    # Read and process the input text in chunks
    with open(input_file, 'r', encoding='utf-8') as f:
        while True:
            # Read a chunk of the input text
            text_chunk = f.read(max_chunk_size)
            
            # If the chunk is empty, break from the loop
            if not text_chunk:
                break
            
            # Restore punctuation for the current chunk
            punctuated_chunk = model.restore_punctuation(text_chunk)
            
            # Append the punctuated chunk to the list
            punctuated_text_chunks.append(punctuated_chunk)

    # Combine all punctuated chunks into a single string
    punctuated_text = ''.join(punctuated_text_chunks)

    # Write the punctuated text to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(punctuated_text)

# Specify input and output file paths
input_file_path = r"text2.txt"  ## Specify the path to your input folder and file
output_file_path = r"expected_output.txt"  # Specify the path to your output folder and file

# Perform punctuation restoration and save the result to the output file
restore_punctuation(input_file_path, output_file_path)
print("Punctuation restoration completed. Punctuated text saved to", output_file_path)

