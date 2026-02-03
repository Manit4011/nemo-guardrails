import re
from typing import Any, List, Literal, Optional, Union

class TextSplitter:
    def __init__(self, chunk_size: int=512, chunk_overlap: int=32, length_function=len, keep_separator: bool=False):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator

    def split_text(self, text: str) -> List[str]:
        raise NotImplementedError

class RecursiveCharacterTextSplitter(TextSplitter):
    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = self._split_text_with_regex(text, _separator, self._keep_separator)
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, self._separators)

    @staticmethod
    def _split_text_with_regex(
        text: str, separator: str, keep_separator: Union[bool, Literal["start", "end"]]
    ) -> List[str]:
        if separator:
            if keep_separator:
                _splits = re.split(f"({separator})", text)
                splits = (
                    ([_splits[i] + _splits[i + 1] for i in range(0, len(_splits) - 1, 2)])
                    if keep_separator == "end"
                    else ([_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)])
                )
                if len(_splits) % 2 == 0:
                    splits += _splits[-1:]
                splits = (
                    (splits + [_splits[-1]])
                    if keep_separator == "end"
                    else ([_splits[0]] + splits)
                )
            else:
                splits = re.split(separator, text)
        else:
            splits = list(text)
        return [s for s in splits if s != ""]

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        chunks = []
        current_chunk = splits[0]
        for split in splits[1:]:
            if len(current_chunk) + len(split) + len(separator) <= self._chunk_size:
                current_chunk += separator + split
            else:
                chunks.append(current_chunk)
                current_chunk = split
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

class SentenceTokenTextSplitter(TextSplitter):
    def __init__(self, chunk_size: int=512, chunk_overlap: int=32, length_function=len):
        super().__init__(chunk_size, chunk_overlap, length_function)
    
    def split_text(self, text: str) -> List[str]:
        # Splitting the text into sentences using regex
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = self._length_function(sentence)
            if current_length + sentence_length > self._chunk_size:
                # If the chunk exceeds the limit, finalize the current chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            # Add the last chunk
            chunks.append(" ".join(current_chunk))

        # Handle overlap by repeating sentences at the boundary
        if self._chunk_overlap > 0:
            chunks = self._add_overlap(chunks)

        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlapping between chunks."""
        overlap_chunks = []
        overlap_text = ""
        for chunk in chunks:
            combined_chunk = overlap_text + chunk
            overlap_chunks.append(combined_chunk)
            overlap_text = " ".join(chunk.split()[-self._chunk_overlap:])
        return overlap_chunks