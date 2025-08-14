def classify_document(text):
    """
    Classifies the document content based on its text.
    This classification is based on specific keywords found in the text.
    Modify the keywords and conditions according to your specific needs.

    :param text: The text content of the document
    :return: A string indicating the document's classification
    """
    # Convert text to lowercase for case-insensitive comparison
    text = text.lower()
    
    # Check for specific keywords to classify the document
    if "course content" in text:
        return "Course Content Document"
    elif "assessment" in text:
        return "Assessment Document"
    elif "syllabus" in text:
        return "Syllabus Document"
    elif "grading" in text:
        return "Grading Document"
    else:
        return "Unclassified Document"
