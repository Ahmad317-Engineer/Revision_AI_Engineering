def summarize(filepath):
    """
    Reads a text file and returns a dictionary containing
    the total word count and the text of the first line.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    if not lines:
        return {
            "word_count": 0,
            "first_line": ""
        }
    
    first_line = lines[0].strip()
    
    full_text = "".join(lines)
    word_count = len(full_text.split())
    
    return {
        "word_count": word_count,
        "first_line": first_line
    }

file_to_check = "sample_notes.txt"

try:
    summary = summarize(file_to_check)
    print(f"Analysis for '{file_to_check}':")
    print(f" - Word Count: {summary['word_count']}")
    print(f" - First Line: \"{summary['first_line']}\"")
except FileNotFoundError:
    print(f"Error: The file '{file_to_check}' was not found. Please check the path.")