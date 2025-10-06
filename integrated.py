import os
import json
import random
import re
import requests
import ast
import operator as op


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def _ensure_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')


class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ChatbotAssistant:
    def __init__(self, intents_path, function_mappings=None):
        self.model = None
        self.intents_path = intents_path

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}

        self.function_mappings = function_mappings or {}

        self.X = None
        self.y = None

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]
        return words

    def bag_of_words(self, words):
        vocab_set = set(words)
        return [1 if word in vocab_set else 0 for word in self.vocabulary]

    def parse_intents(self):
        if not os.path.exists(self.intents_path):
            raise FileNotFoundError(
                f"Could not find intents file: {self.intents_path}")

        with open(self.intents_path, 'r', encoding='utf-8') as f:
            intents_data = json.load(f)

        for intent in intents_data['intents']:
            tag = intent['tag']
            if tag not in self.intents:
                self.intents.append(tag)
                self.intents_responses[tag] = intent.get('responses', [])

            for pattern in intent.get('patterns', []):
                pattern_words = self.tokenize_and_lemmatize(pattern)
                self.vocabulary.extend(pattern_words)
                self.documents.append((pattern_words, tag))

        self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = []
        indices = []

        for pattern_words, tag in self.documents:
            bag = self.bag_of_words(pattern_words)
            intent_index = self.intents.index(tag)
            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags, dtype=np.float32)
        self.y = np.array(indices, dtype=np.int64)

    def train_model(self, batch_size=8, lr=0.01, epochs=100):
        if self.X is None or self.y is None:
            raise RuntimeError("Call prepare_data() before training.")

        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / max(1, len(loader))
            # print(f"Epoch {epoch + 1}: loss: {avg_loss:.4f}")

    def save_model(self, model_path, dimensions_path):
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        torch.save(self.model.state_dict(), model_path)
        with open(dimensions_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'input_size': self.X.shape[1],
                    'output_size': len(self.intents)
                },
                f
            )

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r', encoding='utf-8') as f:
            dimensions = json.load(f)
        self.model = ChatbotModel(
            dimensions['input_size'], dimensions['output_size'])
        state = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state)
        self.model.eval()

    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        if predicted_intent in self.function_mappings:
            try:
                result = self.function_mappings[predicted_intent](
                    input_message)
                return f"{random.choice(self.intents_responses.get(predicted_intent, ['']))} {result}"
            except Exception as e:
                print(f"Function mapping error for '{predicted_intent}': {e}")

        responses = self.intents_responses.get(predicted_intent, [])
        if responses:
            return random.choice(responses)
        return "I'm sorry, I don't understand that."


# -------------------------
# Extra functions
# -------------------------
def get_joke(_msg=None):
    try:
        response = requests.get(
            "https://v2.jokeapi.dev/joke/Any?type=single", timeout=5)
        if response.status_code != 200:
            print(f"JokeAPI returned status {response.status_code}")
            return "Sorry, I couldn’t fetch a joke right now."
        data = response.json()
        return data.get("joke", "Sorry, I couldn’t fetch a joke right now.")
    except Exception as e:
        print(f"Joke fetch error: {e}")
        return "Oops! Something went wrong fetching a joke."


def get_stocks(_msg=None):
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA',
              'BRK.A', 'V', 'JPM', 'JNJ', 'WMT', 'PG', 'UNH', 'DIS', 'MA', 'HD', 'PYPL']
    return random.sample(stocks, 3)


def get_weather(city="London"):
    API_KEY = "YOUR_OPENWEATHER_API_KEY"  # insert your key here
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url).json()
        if response.get("main"):
            temp = response["main"]["temp"]
            desc = response["weather"][0]["description"]
            return f"The weather in {city} is {desc} with {temp}°C. More details: https://www.weather.com/"
        else:
            return " Check https://www.weather.com/"
    except Exception as e:
        return f"Error fetching weather: {e}"


def get_weather_from_message(message):
    match = re.search(r"weather in ([a-zA-Z\s]+)", message.lower())
    city = match.group(1).strip().title() if match else "London"
    return get_weather(city)


def get_news(_msg=None):
    API_KEY = "YOUR_NEWSAPI_KEY"  # insert your key here
    url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={API_KEY}"
    try:
        response = requests.get(url).json()
        articles = response.get("articles", [])[:3]
        if not articles:
            return " Check https://news.google.com/"
        return " | ".join([a["title"] for a in articles]) + " — more at https://news.google.com/"
    except Exception as e:
        return f"Error fetching news: {e}"


# Safe operators allowed
operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod
}


def safe_eval(expr):
    """
    Safely evaluate a math expression string like '2+3*4'
    """
    try:
        node = ast.parse(expr, mode='eval').body

        def _eval(node):
            if isinstance(node, ast.Num):  # numbers
                return node.n
            elif isinstance(node, ast.BinOp):  # operations
                return operators[type(node.op)](_eval(node.left), _eval(node.right))
            else:
                raise TypeError("Unsupported expression")
        return _eval(node)
    except Exception:
        return "Sorry, I couldn’t calculate that."


def calculator(msg=None):
    if not msg:
        return "Please provide an expression to calculate."
    # Extract numbers & operators from user message
    expr = msg.lower().replace("calculate", "").replace("solve", "").strip()
    expr = expr.replace("what is", "").strip()
    return safe_eval(expr)


# -------------------------
# Main
# -------------------------
if __name__ == '__main__':
    _ensure_nltk()
    assistant = ChatbotAssistant(
        'intents.json',
        function_mappings={
            'stocks': get_stocks,
            'weather': lambda msg=None: get_weather_from_message(msg),
            'news': get_news,
            'jokes': get_joke,
            'calculator': calculator
        }
    )

    assistant.parse_intents()
    assistant.prepare_data()

    # 🔎 Auto-detect if retraining is needed
    needs_retrain = True
    if os.path.exists('chatbot_model.pth') and os.path.exists('dimensions.json'):
        with open('dimensions.json', 'r', encoding='utf-8') as f:
            saved_dims = json.load(f)
        current_input_size = assistant.X.shape[1]
        current_output_size = len(assistant.intents)

        if (saved_dims['input_size'] == current_input_size and
                saved_dims['output_size'] == current_output_size):
            needs_retrain = False

    if needs_retrain:
        print("Training new model (intents changed or no saved model)...")
        assistant.train_model(batch_size=8, lr=0.01, epochs=100)
        assistant.save_model('chatbot_model.pth', 'dimensions.json')
    else:
        print("Loading existing model...")
        assistant.load_model('chatbot_model.pth', 'dimensions.json')

    # Chat loop
    while True:
        message = input("Enter your message: ")
        if message == '/quit':
            break
        print(assistant.process_message(message))
