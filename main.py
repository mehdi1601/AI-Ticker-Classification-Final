# Mehdi Dahlouk
# Class: C964 Capstone Task 2
# Program Name: AI Ticket Classifier

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import os  # For directory operations
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud  
from sklearn.model_selection import learning_curve

# Initialize Flask app
app = Flask(__name__)

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Text Preprocessing Setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text.lower())
    # Lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

def classify_ticket_type(text):
        text = text.lower()
        # Enhanced Training with Keywords to improve accuracy (on top of the existing dataset)
        billing_keywords = [
            "bill", "charge", "payment", "invoice", "fees", "charges", "billing",
            "account charge", "subscription fee", "statement", "billing issue", "overcharge",
            "billing problem", "financial", "cost", "price", "expense", "pay", "due", "overbilling",
            "auto-pay", "autopay", "billing cycle", "monthly charge", "balance", "billing query",
            "billing dispute", "chargeback", "credit", "debit", "financial query", "payment issue",
            "payment problem", "pay invoice", "pay bill", "direct debit", "financial assistance",
            "payment method", "payment option", "payment plan", "pricing", "rate", "renewal",
            "subscription", "tariff", "transaction", "billing adjustment", "billing error",
            "charge dispute", "credit card", "payment failure", "payment error",
            "account balance", "payment confirmation", "late fee", "service charge",
            "payment processing", "payment terms", "payment history", "billing contact",
            "automated billing", "recurring charge", "bilb ling support", "payment receipt",
            "payment deadline", "outstanding balance", "billing cycle end", "pro-rated charge",
            "billing statement", "annual fee", "quarterly billing", "monthly billing",
            "billing account", "payment dispute", "billing clarification", "e-billing",
            "online payment", "secure payment", "payment gateway", "billing notification",
            "payment reminder", "past due", "arrears", "financial statement", "account statement",
            "payment status", "unauthorized charge", "billing adjustment request", "payment extension",
            "payment deferment", "billing cycle date", "billing frequency", "invoice number",
            "invoice date", "invoice amount", "payment breakdown", "itemized bill", "billing inquiry"
        ]
        cancellation_keywords = [
            "cancel", "terminate", "unsubscribe", "stop", "end", "revoke", "abort", "discontinue",
            "cancel order", "cancel subscription", "cancel service", "cancel booking", "cancel reservation",
            "cancel membership", "cancel account", "cancel application", "cancel registration",
            "halt", "suspend", "withdraw", "rescind", "annul", "cease", "nullify", "void",
            "cancelation request", "termination request", "discontinuation", "opt out", "drop out",
            "break off", "call off", "cut off", "delete", "erase", "remove", "undo", "cancel policy",
            "cancellation policy", "cancellation fee", "cancellation charges", "no cancel",
            "cannot cancel", "unable to cancel", "cancel immediately", "urgent cancellation",
            "expedite cancellation", "cancel confirmation", "confirm cancellation", "cancellation confirmed",
            "cancel without penalty", "cancel without charge", "cancel with refund", "early cancellation",
            "late cancellation", "cancel due to", "reason for cancellation", "cancellation conditions"
        ]
        product_keywords = [
            "product", "item", "purchase", "order", "buy", "goods", "merchandise", "product info",
            "product details", "product query", "product question", "product issue", "product information",
            "specifications", "features", "availability", "stock", "in stock", "out of stock",
            "backorder", "release date", "new product", "latest product", "upcoming product",
            "product comparison", "compare products", "product range", "product selection",
            "product catalog", "catalogue", "product line", "product model", "model number",
            "product version", "product upgrade", "accessories", "product accessories",
            "product quality", "warranty", "guarantee", "product support", "customer support",
            "product guide", "user guide", "manual", "instruction manual", "product instructions",
            "usage", "how to use", "how to operate", "product operation", "product troubleshooting",
            "product help", "product assistance", "product advice", "recommend product",
            "product recommendation", "best product", "top product", "product review", "product feedback"
        ]
        refund_keywords = refund_keywords = [
            "refund", "return", "money back", "reimbursement", "exchange", "repayment", "compensation",
            "refund policy", "refund request", "refund status", "refund process", "refund query",
            "refund issue", "refund procedure", "refund eligibility", "refund terms", "refund conditions",
            "return process", "return request", "return policy", "return item", "return order",
            "cancel order", "cancel purchase", "reverse payment", "reverse charge", "refund transaction",
            "refund amount", "refund confirmation", "refund application", "refund approval", "refund denied",
            "return goods", "return merchandise", "return product", "refund for return", "refund window",
            "refund period", "refund deadline", "refund timeframe", "refund method", "refund option",
            "full refund", "partial refund", "no refund", "chargeback", "dispute charge",
            "cancel transaction", "revoke payment", "refund dispute", "refund claim", "refund appeal"
        ]
        technical_keywords = [
            "problem", "issue", "not working", "crash", "error", "bug", "fault", "glitch", "technical",
            "defect", "malfunction", "broken", "faulty", "trouble", "difficulty", "complication",
            "hindrance", "obstacle", "setback", "snag", "not starting", "not working right",
            "crashing", "failure", "system down", "unresponsive", "freeze", "frozen", "hang", "stuck",
            "slow", "performance issue", "connectivity problem", "network issue", "connection issue",
            "downtime", "outage", "disruption", "interruption", "corruption", "data loss",
            "sync issue", "synchronization problem", "login issue", "access problem", "authentication issue",
            "security breach", "vulnerability", "exploit", "hack", "cyber attack", "malware", "virus",
            "spyware", "adware", "trojan", "worm", "ransomware", "phishing", "spam", "scam", "fraud"
        ]
        if any(keyword in text for keyword in billing_keywords):
            return "Billing Inquiry"
        elif any(keyword in text for keyword in cancellation_keywords):
            return "Cancellation Request"
        elif any(keyword in text for keyword in product_keywords):
            return "Product Inquiry"
        elif any(keyword in text for keyword in refund_keywords):
            return "Refund Request"
        elif any(keyword in text for keyword in technical_keywords):
            return "Technical Issue"
        else:
            return "Unknown"

# Load and preprocess dataset
df = pd.read_csv('data_set/customer_support_tickets.csv')
df['Ticket Description'] = df['Ticket Description'].astype(str).apply(preprocess_text)


# Selecting features and target variable
X = df['Ticket Description']
y = df['Ticket Type']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create the pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# Hyperparameter grid
param_grid = {
    'tfidf__ngram_range': [(1, 2), (1, 3)],
    'classifier__max_iter': [100, 200, 500]
}

# Grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)


# Function to generate word cloud from ticket descriptions
def generate_word_cloud(text_data):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    wordcloud.to_file('static/wordcloud.png')  # Save the word cloud as an image file
    return 'wordcloud.png'

def generate_ticket_length_histogram(df):
    if not os.path.exists('static'):
        os.makedirs('static')

    ticket_lengths = df['Ticket Description'].str.split().apply(len)
    plt.figure(figsize=(10, 6))
    plt.hist(ticket_lengths, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Ticket Lengths')
    plt.xlabel('Ticket Length (Number of Words)')
    plt.ylabel('Frequency')

    histogram_file = 'static/ticket_length_histogram.png'
    plt.savefig(histogram_file)
    plt.clf()

    return histogram_file
# Function to generate Learning Curve for our Analysis
def generate_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score', color='blue')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.1)
    plt.plot(train_sizes, test_mean, label='Cross-validation score', color='orange')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='orange', alpha=0.1)
    plt.title('Learning Curve')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    learning_curve_file = 'static/learning_curve.png'
    plt.savefig(learning_curve_file)
    plt.clf()

    return learning_curve_file

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['ticket_text']
        predicted_category = classify_ticket_type(user_input)
        return render_template('result.html', category=predicted_category, ticket_text=user_input)
    return render_template('index.html')

@app.route('/wordcloud')
def wordcloud():
    # Generate word cloud from the ticket descriptions
    text_data = ' '.join(df['Ticket Description'])  # Concatenate all ticket descriptions into a single string
    wordcloud_image = generate_word_cloud(text_data)

    # Pass the filename of the word cloud image to the template
    return render_template('index.html', wordcloud_image=wordcloud_image)

@app.route('/error_graph')
def error_graph():
    # Generate the graph and get the file path
    graph_image = generate_error_graph()

    # Extract the filename for use in the template
    filename = graph_image.split('/')[-1]

    # Render a template and pass the filename of the graph image
    return render_template('show_graph.html', graph_image=filename)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)
