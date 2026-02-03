from utils.split_functions import RecursiveCharacterTextSplitter #, SentenceTokenTextSplitter

def create_chunks(text: str, chunkSize: int=512, overlap: int=32):
    textSplitter = RecursiveCharacterTextSplitter(
      chunk_size=chunkSize,
      chunk_overlap=overlap,
      length_function=len,
      is_separator_regex=False
    )
    # textSplitter = SentenceTokenTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = textSplitter.split_text(text)
    return chunks