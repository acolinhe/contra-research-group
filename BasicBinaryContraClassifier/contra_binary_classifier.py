import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

lemmatizer = WordNetLemmatizer()


def load_model_if_exists(model_name):
    """Load a model if it exists, otherwise return None."""
    if os.path.exists(model_name):
        return joblib.load(model_name)
    return None


def evaluate_model(pipeline, X_train, y_train, X_test, y_test, model_name):
    """Train model and evaluate it on the test set, then save the model."""
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, model_name)  # Save the model
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")


def example_contradictions():
    contra_examples = [['I love dogs!', 'I hate dogs!'],
                       ['I really enjoy hiking on the weekends.', "I wouldn't say I have ever enjoyed hiking."],
                       ['Anthony is my best friend.', "I don't think I have a best friend."],
                       ['I have always recommended working while going to school.',
                        'Students should not have to work while taking classes in school'],
                       ['I support coal miners and keeping the coal industry strong!',
                        'Coal is in the past and we need to look to the future of renewable energy.'],
                       ["Bakersfield is the best place to raise a family and live life!",
                        "I wouldn't live in Bakersfield, especially to grow a family in."],
                       ["I hate Italian food and it's not my favorite.",
                        "I'm really craving some Italian food to eat right now."],
                       ["My last job was selling cars at the local Honda dealership.",
                        "I have no experience selling cars."],
                       ["Teachers deserve to be paid better for the benefit they give to society.",
                        "I think teachers make enough and don't deserve better pay."],
                       ["I like the taste of coke.", "I'm not a fan of coke, I prefer Pepsi."]]

    non_contra_examples = [['I love dogs!', 'I love animals!'],
                           ['I really enjoyed the hike up to bishop peak last weekend.',
                            "I would say I have always enjoyed hiking."],
                           ['Anthony is my best friend.', "I have one best friend."],
                           ['I have always recommended working while going to school.',
                            'Students should work while taking classes in school to gain real world experience.'],
                           ['I support coal miners and keeping the coal industry strong!',
                            "Coal is reliable and we can't rely on unpredictable renewable energy."],
                           ["Bakersfield is the best place to raise a family and live life!",
                            "Bakersfield is cheap to live in and family oriented, making it great to raise my family "
                            "in."],
                           ["I hate Italian food and it's not my favorite.",
                            "I really love eating Italian dishes like pasta and lasagna."],
                           ["My last job was selling cars at the local Honda dealership.",
                            "I remember selling a new Honda civic last month."],
                           ["Teachers deserve to be paid better for the benefit they give to society.",
                            "Teachers are really important to society and I hope they make more soon."],
                           ["I like the taste of Coke.", "I'm a big fan of soda, especially Coke."]]
    return contra_examples, non_contra_examples


def preprocess_text(df):
    """Preprocess text by tagging, tokenizing, lowercasing, and lemmatizing."""
    df['combined_text'] = df.apply(lambda row: ' '.join(apply_tags_and_process(row['text1'], row['text2'])), axis=1)
    return df


def apply_tags_and_process(text1, text2):
    tagged_text1 = ['s1-' + token for token in word_tokenize(text1)]
    tagged_text2 = ['s2-' + token for token in word_tokenize(text2)]
    combined_tagged_text = tagged_text1 + tagged_text2
    processed_tokens = [lemmatizer.lemmatize(token.lower()) for token in combined_tagged_text]
    return processed_tokens


def convert_labels(df):
    """Convert labels from multi-class to binary class system."""
    # 0 non contra, 1 contra
    df['label'] = df['label'].replace({1: 0, 2: 1})


def make_predictions(nb_model, sentence_pairs):
    results = []

    for sentence1, sentence2 in sentence_pairs:
        combined_text = ' '.join(
            apply_tags_and_process(sentence1, sentence2) + apply_tags_and_process(sentence2, sentence1))

        nb_prediction = nb_model.predict([combined_text])[0]

        results.append({
            "Sentence 1": sentence1,
            "Sentence 2": sentence2,
            "Naive Bayes Prediction": nb_prediction,
            "Random Forest Prediction": None
        })

    return results


def swap_and_concatenate(df, col1, col2):
    df_swapped = df.copy()
    df_swapped[col1], df_swapped[col2] = df[col2], df[col1]
    df_combined = pd.concat([df, df_swapped], ignore_index=True)
    return df_combined


def load_and_prepare_data(train_file, validation_file):
    df = pd.read_json(train_file, lines=True)
    validation_matched = pd.read_json(validation_file, lines=True)
    df = pd.concat([df, validation_matched], ignore_index=True)
    df = df.drop(['idx', 'label_text'], axis=1)
    return df


def create_pipelines():
    nb_pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])

    rf_pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', RandomForestClassifier(n_jobs=-1))
    ])
    return nb_pipeline, rf_pipeline


def get_nb_feature_importances(nb_model, vectorizer):
    feature_log_probs = nb_model.feature_log_prob_
    feature_names = vectorizer.get_feature_names_out()
    importance_df = pd.DataFrame(feature_log_probs.T, index=feature_names, columns=["Class 0", "Class 1"])
    importance_df['Importance'] = importance_df.max(axis=1)
    return importance_df.sort_values(by='Importance', ascending=False)


def get_rf_feature_importances(rf_model, vectorizer):
    feature_importances = rf_model.feature_importances_
    feature_names = vectorizer.get_feature_names_out()
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    return importance_df.sort_values(by='Importance', ascending=False)


def main():
    contra_examples, non_contra_examples = example_contradictions()
    correct_nb_contra, correct_rf_contra = 0, 0
    correct_nb_non, correct_rf_non = 0, 0

    df = load_and_prepare_data('data/MNLI/train1.jsonl', 'data/MNLI/train2.jsonl')
    train, test = train_test_split(df, test_size=0.3, random_state=42)

    train = swap_and_concatenate(train, 'text1', 'text2')
    test = swap_and_concatenate(test, 'text1', 'text2')

    train = preprocess_text(train)
    test = preprocess_text(test)

    convert_labels(train)
    convert_labels(test)

    train.to_csv('train_df.csv', index=False)
    test.to_csv('test_df.csv', index=False)

    nb_pipeline, rf_pipeline = create_pipelines()

    nb_model_name = 'nb_model.joblib'
    # rf_model_name = 'rf_model.joblib'

    nb_pipeline = load_model_if_exists(nb_model_name) or nb_pipeline
    # rf_pipeline = load_model_if_exists(rf_model_name) or rf_pipeline

    if not load_model_if_exists(nb_model_name):
        print("Evaluating Naive Bayes on test set")
        evaluate_model(nb_pipeline, train['combined_text'], train['label'], test['combined_text'], test['label'], nb_model_name)

    # if not load_model_if_exists(rf_model_name):
    #     print("Evaluating Random Forest on test set")
    #     evaluate_model(rf_pipeline, train['combined_text'], train['label'], test['combined_text'], test['label'], rf_model_name)

    results = make_predictions(nb_pipeline, contra_examples + non_contra_examples)
    for result in results[:11]:
        if result['Naive Bayes Prediction'] == 1:
            correct_nb_contra += 1
        # if result['Random Forest Prediction'] == 1:
        #     correct_rf_contra += 1

    for result in results[11:]:
        if result['Naive Bayes Prediction'] == 0:
            correct_nb_non += 1
        # if result['Random Forest Prediction'] == 0:
        #     correct_rf_non += 1

    print('Correct Naive Bayes Contra: ', correct_nb_contra / 10)
    # print('Correct Random Forest Contra: ', correct_rf_contra / 10)
    print('Correct Naive Bayes Non-Contra: ', correct_nb_non / 10)
    # print('Correct Random Forest Non-Contra: ', correct_rf_non / 10)

    nb_feature_importances = get_nb_feature_importances(nb_pipeline.named_steps['classifier'], nb_pipeline.named_steps['vectorizer'])
    print("Top features for Naive Bayes:")
    print(nb_feature_importances.head(10))
    print()

    # rf_feature_importances = get_rf_feature_importances(rf_pipeline.named_steps['classifier'], rf_pipeline.named_steps['vectorizer'])
    # print("Top features for Random Forest:")
    # print(rf_feature_importances.head(10))


if __name__ == '__main__':
    main()
