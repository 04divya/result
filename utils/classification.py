def classify_document(text):
    text = text.lower()
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
