import nltk
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def ensurenltkdata():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)
ensurenltkdata()
chatbotresponses = [
    "Hello, how can I assist you?",
    "I'm doing well, thanks for asking.",
    "What is your name?",
    "I am a chatbot designed using NLTK.",
    "What are your capabilities?",
    "I can converse and answer simple questions.",
    "How does NLP work?",
    "NLP works by analyzing and understanding human language through algorithms.",
    "What is NLTK?",
    "NLTK is a toolkit in Python to process and analyze human language.",
    "What is spaCy?",
    "spaCy is an open-source library for advanced NLP tasks.",
    "Goodbye",
    "Goodbye! Have a great day!",
    "Thank you",
    "You're welcome!"
]
def normalizetext(inputtext):
    return inputtext.lower().translate(str.maketrans('', '', string.punctuation))
normalizedresponses = [normalizetext(response) for response in chatbotresponses]
def getchatbotresponse(userinput):
    normalizedinput = normalizetext(userinput)
    normalizedresponses.append(normalizedinput)
    vectorizer = TfidfVectorizer()
    tfidfmatrix = vectorizer.fit_transform(normalizedresponses)
    similaritiescores = cosine_similarity(tfidfmatrix[-1], tfidfmatrix[:-1])
    responseidx = np.argmax(similaritiescores)
    normalizedresponses.pop()
    if similaritiescores[0][responseidx] < 0.3:
        return "Sorry, I didn't understand that. Could you rephrase it?"
    if responseidx % 2 == 0 and responseidx + 1 < len(chatbotresponses):
        return chatbotresponses[responseidx + 1]
    else:
        return chatbotresponses[responseidx]
def runchatbot():
    print("Chatbot: Hello! I'm an NLP-powered chatbot. Type 'quit' to exit.")
    while True:
        userinput = input("You: ")
        if userinput.lower() in ['quit', 'exit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        response = getchatbotresponse(userinput)
        print("Chatbot:", response)
if __name__ == "__main__":
    runchatbot()
