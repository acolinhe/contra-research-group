import pandas as pd
from nltk import download
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def preprocess_text(text):
    """Preprocess text by tokenizing, removing stopwords, and lemmatizing."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def convert_labels(df):
    """Convert labels from multi-class to binary class system."""
    df['label'] = df['label'].replace({1: 0, 2: 1})

def tag_text(df):
    """Create two versions of tagged text with 's1' always starting, and concatenate them as separate rows."""
    df_original = df.copy()
    df_switched = df.copy()

    df_original['combined_text'] = df_original.apply(
        lambda row: ' '.join(['s1-' + word for word in row['text1'].split()]) +
                    ' ' + ' '.join(['s2-' + word for word in row['text2'].split()]), axis=1)

    df_switched['combined_text'] = df_switched.apply(
        lambda row: ' '.join(['s1-' + word for word in row['text2'].split()]) +
                    ' ' + ' '.join(['s2-' + word for word in row['text1'].split()]), axis=1)

    return pd.concat([df_original, df_switched], ignore_index=True)

def evaluate_model(pipeline, X_train, y_train, X_test, y_test):
    """Train model and evaluate it on the test set."""
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

def make_predictions(nb_model, rf_model, sentence_pairs):
    results = []

    for sentence1, sentence2 in sentence_pairs:
        combined_text = ' '.join(['s1-' + word for word in sentence1.split()]) + \
                        ' ' + ' '.join(['s2-' + word for word in sentence2.split()])

        nb_prediction = nb_model.predict([combined_text])[0]
        rf_prediction = rf_model.predict([combined_text])[0]

        results.append({
            "Sentence 1": sentence1,
            "Sentence 2": sentence2,
            "Naive Bayes Prediction": nb_prediction,
            "Random Forest Prediction": rf_prediction
        })

    return results

def main():
    contra_examples = [['I love dogs!', 'I hate dogs!'],
     ['I really enjoy hiking on the weekends.', "I wouldn't say I have ever enjoyed hiking."],
     ['Anthony is my best friend.', "I don't think I have a best friend."],
     ['I have always recommended working while going to school.', 'Students should not have to work while taking classes in school'],
     ['I support coal miners and keeping the coal industry strong!', 'Coal is in the past and we need to look to the future of renewable energy.'],
     ["Bakersfield is the best place to raise a family and live life!", "I wouldn't live in Bakersfield, especiaally to grow a family in."],
     ["I hate Italian food and it's not my favorite.", "I'm really craving some Italian food to eat right now."],
     ["My last job was selling cars at the local Honda dealership.", "I have no experience selling cars."],
     ["Teachers deserve to be paid better for the benefit they give to society.", "I think teachers make enough and don't deserve better pay."],
     ["I like the taste of coke.", "I'm not a fan of coke, I prefer Pepsi."]
     ]

    df = pd.read_json('data/train1.jsonl', lines=True)
    validation_matched = pd.read_json('data/train2.jsonl', lines=True)
    df = pd.concat([df, validation_matched], ignore_index=True)

    train, test = train_test_split(df, test_size=0.3, random_state=42)
    train = tag_text(train)
    test = tag_text(test)

    convert_labels(train)
    convert_labels(test)

    nb_pipeline = Pipeline([
        ('vectorizer', CountVectorizer(preprocessor=preprocess_text)),
        ('classifier', MultinomialNB())
    ])

    rf_pipeline = Pipeline([
        ('vectorizer', CountVectorizer(preprocessor=preprocess_text)),
        ('classifier', RandomForestClassifier(n_jobs=-1))
    ])

    print("Evaluating Naive Bayes on test set")
    evaluate_model(nb_pipeline, train['combined_text'], train['label'], test['combined_text'], test['label'])

    print("Evaluating Random Forest on test set")
    evaluate_model(rf_pipeline, train['combined_text'], train['label'], test['combined_text'], test['label'])

    results = make_predictions(nb_pipeline, rf_pipeline, contra_examples)
    for result in results:
        print(result)

if __name__ == '__main__':
    main()
