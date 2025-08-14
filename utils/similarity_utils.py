from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize SentenceTransformer model and NLTK tools
model = SentenceTransformer('all-mpnet-base-v2')
stop_words = set(stopwords.words('english'))
stop_words_minimal = set(stopwords.words('english')) - {
    'database', 'sql', 'data', 'system', 'model', 'query', 'normalization', 
    'entity', 'relationship', 'design', 'table', 'index', 'key', 'programming', 
    'computer', 'software', 'network', 'algorithm', 'technology', 'machine', 
    'learning', 'science', 'engineering', 'artificial', 'intelligence'
}

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Preprocess text: lowercase, remove punctuation, stopwords, and lemmatize."""
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    tokens = word_tokenize(text)
    
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)


bilingual_dict = {
    r'\bpangkalan\s*data\b': 'database',
    r'\bpenormalan\b': 'normalization',
    r'\bpertanyaan\b': 'query',
    r'\bhubungan\s*entiti\b': 'entity relationship',
    r'\bsistem\b': 'system',
    r'\bdata\b': 'data',
    r'\bsql\b': 'sql',
    r'\bmodel\b': 'model',
    r'\brekabentuk\b': 'design',
    r'\bjadual\b': 'table',
    r'\bindeks\b': 'index',
    r'\bkunci\b': 'key',
    r'\bentiti\b': 'entity',
    r'\brelasi\b': 'relationship',
    r'\bpengaturcaraan\b': 'programming',
    r'\bkomputer\b': 'computer',
    r'\bperisian\b': 'software',
    r'\brangkaian\b': 'network',
    r'\balkhwarizmi\b': 'algorithm',
    r'\bteknologi\b': 'technology',
    r'\bpembelajaran\s*mesin\b': 'machine learning',
    r'\bkecerdasan\s*buatan\b': 'artificial intelligence',
    r'\bsains\s*komputer\b': 'computer science',
    r'\bkejuruteraan\b': 'engineering',
    r'\bpenghayatan\s*etika\b': 'ethics appreciation',
    r'\bprojek\b': 'project',
    r'\blatihan\s*industri\b': 'industrial training',
    r'\bmatematik\s*diskret\b': 'discrete mathematics',
    r'\bpengkomputeran\b': 'computing'
}

def translate_malay_to_english(text):
    text = text.lower()
    for malay, english in bilingual_dict.items():
        text = re.sub(malay, english, text, flags=re.IGNORECASE)
    return text

def custom_tokenizer(text):
    tokens = []
    words = re.split(r'\s+', text)
    i = 0
    while i < len(words):
        if re.match(r'[A-Z]{4}\d{4}', words[i]):
            tokens.append(words[i])
            i += 1
        elif i + 1 < len(words) and f"{words[i]} {words[i+1]}" in [v for k, v in bilingual_dict.items()]:
            tokens.append(f"{words[i]} {words[i+1]}")
            i += 2
        elif i + 2 < len(words) and f"{words[i]} {words[i+1]} {words[i+2]}" in [v for k, v in bilingual_dict.items()]:
            tokens.append(f"{words[i]} {words[i+1]} {words[i+2]}")
            i += 3
        else:
            tokens.append(words[i])
            i += 1
    return [t for t in tokens if t]

def preprocess_texts(text, minimal=False):
    if not text or not isinstance(text, str):
        return ""
    text = translate_malay_to_english(text)
    text = text.lower()
    text = re.sub(r'[^\w\s\d]', ' ', text)
    if minimal:
        return text.strip()
    tokens = custom_tokenizer(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words_minimal and len(token) > 1]
    return ' '.join(tokens) if tokens else text.strip()

def calculate_bert_similarity(text1, text2):
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    
    if not text1.strip() or not text2.strip():
        return 0.0
    
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2)
    score = float(similarity.item()) * 100

    if score < 80:
        text1 = preprocess_texts(text1, minimal=True)
        text2 = preprocess_texts(text2, minimal=True)

        if not text1.strip() or not text2.strip():
            return 0.0
        emb1 = model.encode(text1, convert_to_tensor=True)
        emb2 = model.encode(text2, convert_to_tensor=True)
        similarity2 = util.cos_sim(emb1, emb2)
        score2 = float(similarity2.item()) * 100

        score = max(score, score2)
    
    return score

def calculate_tfidf_similarity(text1, text2):
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    
    if not text1.strip() or not text2.strip():
        print("Warning: One or both preprocessed texts are empty")
        return 0.0
    
    vectorizer = TfidfVectorizer(
        max_df=1.0,          
        min_df=1,             
        ngram_range=(1, 4),   
        sublinear_tf=True,    
        max_features=2000,    
        token_pattern=r'(?u)\b\w+\b'  
    )
    try:
        vectors = vectorizer.fit_transform([text1, text2])
        score = 100 * (1 - cosine_similarity(vectors[0], vectors[1])[0][0])
    except ValueError as e:
        return 0.0
    
    return float(score)