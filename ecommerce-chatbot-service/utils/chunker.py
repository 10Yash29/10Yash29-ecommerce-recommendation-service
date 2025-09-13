import re


def chunk_text(text, max_tokens=100):
    # Basic splitting based on newlines and punctuation
    raw_chunks = re.split(r"(?<=[.?!])\s+", text.strip())

    final_chunks = []
    current_chunk = []

    for sentence in raw_chunks:
        current_chunk.append(sentence)
        if len(" ".join(current_chunk).split()) > max_tokens:
            final_chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        final_chunks.append(" ".join(current_chunk))

    return final_chunks
