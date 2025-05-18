import ftfy


def clean_text(text):
    if not text:
        return ""
    # Fix mojibake issues
    fixed = ftfy.fix_text(text)
    # Replace non-breaking spaces and multiple spaces with a single normal space
    cleaned = fixed.replace("\xa0", " ").replace("\u200b", "").strip()
    # Also normalize multiple spaces to single space
    cleaned = " ".join(cleaned.split())
    return cleaned
