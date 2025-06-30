import nltk
import random
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# --- NLTK Data Downloads ---
# Run these lines once to download necessary NLTK data.
# If you run this script for the first time, uncomment these lines.
nltk.download('punkt')        # For tokenization
nltk.download('wordnet')     # For lemmatization
nltk.download('omw-1.4')     # Open Multilingual Wordnet (required for WordNetLemmatizer)
nltk.download('stopwords')   # For removing common words
nltk.download('punkt_tab')

# --- Initialize Lemmatizer ---
lemmatizer = WordNetLemmatizer()

# --- Define Greetings, Goodbyes, and Fallback Responses ---
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "hello", "I am glad! You are talking to me"]

GOODBYE_INPUTS = ("bye", "goodbye", "see you later", "farewell", "quit")
GOODBYE_RESPONSES = ["Goodbye! Have a great day!", "See you later!", "Farewell! It was nice chatting."]

FALLBACK_RESPONSES = [
    "I'm sorry, I don't understand that.",
    "Could you please rephrase that?",
    "I'm still learning. Can you try asking something else?",
    "That's an interesting query, but I don't have an answer for it right now."
]

# --- Knowledge Base for Generic Queries ---
# This is a simple rule-based system. Each key is a list of keywords,
# and the value is a list of possible responses.
# The chatbot will try to find if any of the keywords are present in the user's input.
KNOWLEDGE_BASE = {
    ("how are you", "how are you doing"): [
        "I'm just a program, but I'm functioning perfectly!",
        "As an AI, I don't have feelings, but I'm ready to assist you!",
        "All good here! How can I help you?"
    ],
    ("what is your name", "your name"): [
        "I don't have a name, I'm a chatbot.",
        "You can call me Chatbot.",
        "I am an AI assistant, designed to help you."
    ],
    ("who created you", "who made you"): [
        "I was created by a large language model.",
        "I am a product of Google.",
        "My creators are the engineers at Google."
    ],
    ("what can you do", "your capabilities", "help"): [
        "I can answer generic questions based on my knowledge base.",
        "I can chat with you and provide information on various topics I've been trained on.",
        "Ask me anything general, and I'll do my best to respond!"
    ],
    ("weather", "temperature"): [
        "I cannot provide real-time weather information as I am not connected to live data feeds.",
        "For current weather updates, please check a dedicated weather application."
    ],
    ("time", "current time"): [
        "I do not have access to real-time clock data.",
        "I cannot tell the exact current time."
    ],
    ("thank you", "thanks"): [
        "You're welcome!",
        "No problem!",
        "Glad I could help!",
        "Happy to assist!"
    ]
}

# --- Text Preprocessing Function ---
def preprocess_text(text):
    """
    Tokenizes, lowercases, removes punctuation, and lemmatizes text.
    """
    # Remove punctuation
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])

    # Tokenize
    words = nltk.word_tokenize(text)

    # Lemmatize and remove stopwords
    processed_words = [
        lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')
    ]
    return processed_words

# --- Response Generation Function ---
def generate_response(user_input):
    """
    Generates a response based on user input.
    """
    processed_input = preprocess_text(user_input)

    # Check for greetings
    for word in processed_input:
        if word in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

    # Check for goodbyes
    for word in processed_input:
        if word in GOODBYE_INPUTS:
            return random.choice(GOODBYE_RESPONSES)

    # Check knowledge base for generic queries
    for keywords, responses in KNOWLEDGE_BASE.items():
        # Check if any keyword from the knowledge base entry is in the processed user input
        if any(keyword in ' '.join(processed_input) for keyword in keywords):
            return random.choice(responses)

    # If no specific match, return a fallback response
    return random.choice(FALLBACK_RESPONSES)

# --- Main Chatbot Loop ---
def start_chatbot():
    """
    Starts the interactive chatbot session.
    """
    print("Chatbot: Hello! I'm a simple AI chatbot. How can I help you today?")
    print("Chatbot: (Type 'bye' or 'quit' to exit)")

    while True:
        user_input = input("You: ").strip()
        if not user_input: # Handle empty input
            continue

        response = generate_response(user_input)
        print(f"Chatbot: {response}")

        # Exit condition
        if any(word in user_input.lower() for word in GOODBYE_INPUTS):
            break

if __name__ == "__main__":
    start_chatbot()