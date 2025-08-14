def classify_document(text):
    text = text.lower()
    if "transcript" in text or "grade" in text or "gpa" in text or "credit" in text:
        return "Student Result / Transcript Document"
    elif "compulsory courses" in text or "programme structure" in text or "credit hours" in text:
        return "Programme Structure Document"
    else:
        return "Unclassified Document"
