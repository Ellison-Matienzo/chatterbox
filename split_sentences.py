import re
from typing import Generator, List, Optional, FrozenSet, Tuple

# A frozenset of common abbreviations to prevent incorrect sentence splitting.
ABBREVIATIONS: FrozenSet[str] = frozenset([
    'dr.', 'mr.', 'mrs.', 'ms.', 'prof.', 'rev.',
    'e.g.', 'i.e.', 'etc.', 'vs.'
])


def _is_sentence_boundary(text: str, match: re.Match) -> bool:
    """
    Determines if a matched terminator is a true sentence boundary.

    Args:
        text: The full text being processed.
        match: The regex match object for a potential sentence terminator.

    Returns:
        True if the terminator is a valid sentence end, False otherwise.
    """
    # Terminators like '?', '!', or '...' are always sentence boundaries.
    if match.group(0) != '.':
        return True

    # A period is not a boundary if part of a URL, number, etc.
    # This is checked by seeing if a non-space character immediately follows.
    end_index = match.end()
    if end_index < len(text) and not text[end_index].isspace():
        return False

    # A period is not a boundary if it's part of a known abbreviation.
    word_start = text.rfind(' ', 0, match.start()) + 1
    preceding_word = text[word_start:end_index].lower()
    if preceding_word in ABBREVIATIONS:
        return False

    return True


def split_into_sentences(text: str) -> Generator[str, None, None]:
    """
    Splits text into sentences, intelligently handling abbreviations and other edge cases.

    Args:
        text: The text to be split.

    Yields:
        Each detected sentence from the text.
    """
    start_index = 0
    # A sentence can end with multiple periods (like an ellipsis) or a single punctuation mark.
    terminator_pattern = r'\.{3,}|[.?!]'

    for match in re.finditer(terminator_pattern, text):
        if _is_sentence_boundary(text, match):
            end_index = match.end()
            sentence = text[start_index:end_index].strip()
            if sentence:
                yield sentence
            start_index = end_index

    # Yield the remaining part of the text if it's not empty.
    remainder = text[start_index:].strip()
    if remainder:
        yield remainder


def _should_merge_lines(
    previous_line: str,
    current_sentence: str,
    next_sentence: Optional[str],
    target_length: int,
    orphan_threshold: int
) -> bool:
    """
    Determines if the current sentence should be merged with the previous line.
    The primary rule is to merge as long as the line length is not exceeded.
    """
    # Only merge if the combined line does not exceed the target length.
    if len(previous_line) + 1 + len(current_sentence) > target_length:
        return False

    # Default to merging if the length rule doesn't prevent it.
    return True


def format_text_into_lines(
    text: str,
    target_length: int = 240,
    orphan_threshold: int = 65
) -> str:
    """
    Splits text into lines by sentence, with rules for abbreviations,
    URLs, and line length optimization.

    Args:
        text: The input text string.
        target_length: The soft maximum target length for a line.
        orphan_threshold: Sentences shorter than this may be appended to a
                          longer line to prevent very short "orphan" lines.

    Returns:
        A single string with sentences formatted into optimized lines.
    """
    # 1. Normalize whitespace and split text into a tuple of sentences.
    normalized_text = ' '.join(text.split())
    sentences: Tuple[str, ...] = tuple(split_into_sentences(normalized_text))

    if not sentences:
        return ""

    # 2. Assemble lines by intelligently merging sentences.
    lines: List[str] = [sentences[0]]
    for i in range(1, len(sentences)):
        current_sentence = sentences[i]
        next_sentence = sentences[i + 1] if i + 1 < len(sentences) else None

        if _should_merge_lines(
            previous_line=lines[-1],
            current_sentence=current_sentence,
            next_sentence=next_sentence,
            target_length=target_length,
            orphan_threshold=orphan_threshold
        ):
            lines[-1] += f" {current_sentence}"
        else:
            lines.append(current_sentence)

    return "\n".join(lines)


def main():
    """Example usage of the text formatter."""
    sample_text = """
    Oooh, it got serious in here.
    Umm...
    How do I get out of that?
    Lets talk about...
    I hate when people interview you, they go...
    you get away with a lot...
    you get away...
    No, I dont! I don't get away with anything because you're...
    you're presuming that what I'm saying is meant in a malicious way.
    It isn't true.
    Jesus, how old do they think that I...
    like...
    I have my jean jacket and my t-shirt on... oh my god.
    """

    formatted_output = format_text_into_lines(sample_text)
    print(formatted_output)


if __name__ == "__main__":
    main()
