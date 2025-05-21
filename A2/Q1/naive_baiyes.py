import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import defaultdict, Counter
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk

# Download required NLTK resources (if not already present)
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# 1. Naïve Bayes Classifier Implementation

class NaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.word_counts = {}         # counts for each class: {class: {word: count}}
        self.total_words = {}         # total word counts for each class
        self.vocab = set()            # global vocabulary
        self.alpha = None             # Laplace smoothing parameter
        self.classes = None           # list of classes

    def fit(self, df, smoothening, class_col="Class Index", text_col="Tokenized Description"):
        """
        Learn model parameters from training data.
        """
        self.alpha = smoothening
        self.classes = df[class_col].unique()
        self.class_priors = {}
        self.word_counts = {c: defaultdict(int) for c in self.classes}
        self.total_words = {c: 0 for c in self.classes}
        doc_counts = {c: 0 for c in self.classes}
        
        for _, row in df.iterrows():
            c = row[class_col]
            doc_counts[c] += 1
            tokens = row[text_col]
            for token in tokens:
                self.word_counts[c][token] += 1
                self.total_words[c] += 1
                self.vocab.add(token)
                
        total_docs = len(df)
        # Using logarithms for priors
        for c in self.classes:
            self.class_priors[c] = np.log(doc_counts[c] / total_docs)
            
        # Precompute log likelihoods for each word in each class
        self.log_likelihoods = {c: {} for c in self.classes}
        V = len(self.vocab)
        for c in self.classes:
            for word in self.vocab:
                count = self.word_counts[c].get(word, 0)
                self.log_likelihoods[c][word] = np.log((count + self.alpha) / (self.total_words[c] + self.alpha * V))
        self.V = V

    def predict(self, df, text_col="Tokenized Description", predicted_col="Predicted"):
        """
        Predicts class labels and appends a column with the predictions.
        """
        predictions = []
        for _, row in df.iterrows():
            tokens = row[text_col]
            class_scores = {}
            for c in self.classes:
                score = self.class_priors[c]
                for token in tokens:
                    if token in self.vocab:
                        score += self.log_likelihoods[c][token]
                    else:
                        # Unseen words: use only smoothing term
                        score += np.log(self.alpha / (self.total_words[c] + self.alpha * self.V))
                class_scores[c] = score
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        df[predicted_col] = predictions
        return df


# 2. Helper Functions for Text Processing

def simple_tokenizer(text):
    """Lowercase, remove punctuation and split on whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    return tokens

def preprocess_tokens(tokens, remove_stopwords=True, stem=True):
    """Removes English stopwords and applies stemming."""
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def create_bigrams(tokens):
    """Creates bigrams from a list of tokens."""
    return [tokens[i] + '_' + tokens[i+1] for i in range(len(tokens)-1)]

def combine_unigrams_bigrams(tokens):
    """Returns a list containing both unigrams and bigrams."""
    return tokens + create_bigrams(tokens)

def add_pos_tags(tokens):
    """Appends POS tag tokens (prefixed with 'POS_') to the token list."""
    tagged = nltk.pos_tag(tokens)
    pos_tokens = ["POS_" + tag for word, tag in tagged]
    return tokens + pos_tokens


# 3. Evaluation and Plotting Functions

def evaluate_model(df, true_col="Class Index", pred_col="Predicted"):
    y_true = df[true_col]
    y_pred = df[pred_col]
    acc = accuracy_score(y_true, y_pred)
    print("Accuracy: {:.2f}%".format(acc * 100))
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_true)))
    plt.xticks(tick_marks, np.unique(y_true))
    plt.yticks(tick_marks, np.unique(y_true))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.show()
    
def plot_wordcloud(freq_dict, title):
    wc = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(freq_dict)
    plt.figure()
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()


# 4. Data Loading (Assuming CSV files)

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Tokenize raw text fields (using simple_tokenizer)
    train_df['Tokenized Description'] = train_df['Description'].apply(simple_tokenizer)
    test_df['Tokenized Description'] = test_df['Description'].apply(simple_tokenizer)
    train_df['Tokenized Title'] = train_df['Title'].apply(simple_tokenizer)
    test_df['Tokenized Title'] = test_df['Title'].apply(simple_tokenizer)
    
    return train_df, test_df


# 5. Main Execution: Running Experiments

if __name__ == '__main__':
    train_path = "../data/Q1/train.csv"
    test_path = "../data/Q1/test.csv"
    
    # Load data
    train_df, test_df = load_data(train_path, test_path)
    
    ########
    # Experiment 1: Naïve Bayes on Raw Description
    ########
    print("=== Experiment 1: Raw Description ===")
    nb_raw = NaiveBayes()
    nb_raw.fit(train_df, smoothening=1.0, class_col="Class Index", text_col="Tokenized Description")
    train_df = nb_raw.predict(train_df, text_col="Tokenized Description", predicted_col="Predicted_Raw")
    test_df = nb_raw.predict(test_df, text_col="Tokenized Description", predicted_col="Predicted_Raw")
    
    print("Training Set Performance (Raw):")
    evaluate_model(train_df, true_col="Class Index", pred_col="Predicted_Raw")
    print("Test Set Performance (Raw):")
    evaluate_model(test_df, true_col="Class Index", pred_col="Predicted_Raw")
    
    # Plot word clouds (most frequent words) for each class from raw model counts
    print("Plotting word clouds for raw description per class...")
    for c in sorted(nb_raw.classes):
        freq_dict = dict(nb_raw.word_counts[c])
        plot_wordcloud(freq_dict, f"Word Cloud for Class {c} (Raw)")
    
    ########
    # Experiment 2: Preprocessing (Stopword Removal + Stemming)
    ########
    print("=== Experiment 2: Preprocessed Description ===")
    # Create new token columns with stopword removal and stemming
    train_df['Tokenized Description_Preproc'] = train_df['Description'].apply(
        lambda x: preprocess_tokens(simple_tokenizer(x)))
    test_df['Tokenized Description_Preproc'] = test_df['Description'].apply(
        lambda x: preprocess_tokens(simple_tokenizer(x)))
    
    # Plot word clouds for each class on preprocessed data
    # (compute frequency counts over the preprocessed tokens for each class)
    class_tokens = {}
    for c in sorted(train_df["Class Index"].unique()):
        tokens = []
        for doc in train_df[train_df["Class Index"]==c]['Tokenized Description_Preproc']:
            tokens.extend(doc)
        class_tokens[c] = dict(Counter(tokens))
        plot_wordcloud(class_tokens[c], f"Word Cloud for Class {c} (Preprocessed)")
    
    # Train Naïve Bayes on preprocessed tokens
    nb_preproc = NaiveBayes()
    nb_preproc.fit(train_df, smoothening=1.0, class_col="Class Index", text_col="Tokenized Description_Preproc")
    train_df = nb_preproc.predict(train_df, text_col="Tokenized Description_Preproc", predicted_col="Predicted_Preproc")
    test_df = nb_preproc.predict(test_df, text_col="Tokenized Description_Preproc", predicted_col="Predicted_Preproc")
    
    print("Training Set Performance (Preprocessed):")
    evaluate_model(train_df, true_col="Class Index", pred_col="Predicted_Preproc")
    print("Test Set Performance (Preprocessed):")
    evaluate_model(test_df, true_col="Class Index", pred_col="Predicted_Preproc")
    
    ########
    # Experiment 3: Unigrams + Bigrams
    ########
    print("=== Experiment 3: Unigrams + Bigrams ===")
    # Create new column combining unigrams and bigrams (based on preprocessed tokens)
    train_df['Tokenized_Desc_UniBi'] = train_df['Tokenized Description_Preproc'].apply(combine_unigrams_bigrams)
    test_df['Tokenized_Desc_UniBi'] = test_df['Tokenized Description_Preproc'].apply(combine_unigrams_bigrams)
    
    nb_unibi = NaiveBayes()
    nb_unibi.fit(train_df, smoothening=1.0, class_col="Class Index", text_col="Tokenized_Desc_UniBi")
    train_df = nb_unibi.predict(train_df, text_col="Tokenized_Desc_UniBi", predicted_col="Predicted_UniBi")
    test_df = nb_unibi.predict(test_df, text_col="Tokenized_Desc_UniBi", predicted_col="Predicted_UniBi")
    
    print("Training Set Performance (Unigrams+Bigrams):")
    evaluate_model(train_df, true_col="Class Index", pred_col="Predicted_UniBi")
    print("Test Set Performance (Unigrams+Bigrams):")
    evaluate_model(test_df, true_col="Class Index", pred_col="Predicted_UniBi")
    
    ########
    # Experiment 5: Title Features Only
    ########
    print("=== Experiment 5a: Raw Title Features ===")
    # Use raw tokenized title (no preprocessing)
    train_df['Tokenized_Title_Raw'] = train_df['Title'].apply(simple_tokenizer)
    test_df['Tokenized_Title_Raw'] = test_df['Title'].apply(simple_tokenizer)
    
    nb_title_raw = NaiveBayes()
    nb_title_raw.fit(train_df, smoothening=1.0, class_col="Class Index", text_col="Tokenized_Title_Raw")
    train_df = nb_title_raw.predict(train_df, text_col="Tokenized_Title_Raw", predicted_col="Predicted_Title_Raw")
    test_df = nb_title_raw.predict(test_df, text_col="Tokenized_Title_Raw", predicted_col="Predicted_Title_Raw")
    
    print("Training Set Performance (Raw Title):")
    evaluate_model(train_df, true_col="Class Index", pred_col="Predicted_Title_Raw")
    print("Test Set Performance (Raw Title):")
    evaluate_model(test_df, true_col="Class Index", pred_col="Predicted_Title_Raw")
    
    
    print("=== Experiment 5b: Preprocessed Title Features ===")
    # Preprocess title texts (stopword removal and stemming)
    train_df['Tokenized_Title_Preproc'] = train_df['Title'].apply(
        lambda x: preprocess_tokens(simple_tokenizer(x)))
    test_df['Tokenized_Title_Preproc'] = test_df['Title'].apply(
        lambda x: preprocess_tokens(simple_tokenizer(x)))
    
    nb_title_preproc = NaiveBayes()
    nb_title_preproc.fit(train_df, smoothening=1.0, class_col="Class Index", text_col="Tokenized_Title_Preproc")
    train_df = nb_title_preproc.predict(train_df, text_col="Tokenized_Title_Preproc", predicted_col="Predicted_Title_Preproc")
    test_df = nb_title_preproc.predict(test_df, text_col="Tokenized_Title_Preproc", predicted_col="Predicted_Title_Preproc")
    
    print("Training Set Performance (Preprocessed Title):")
    evaluate_model(train_df, true_col="Class Index", pred_col="Predicted_Title_Preproc")
    print("Test Set Performance (Preprocessed Title):")
    evaluate_model(test_df, true_col="Class Index", pred_col="Predicted_Title_Preproc")
    
    
    print("=== Experiment 5c: Title Unigrams + Bigrams ===")
    # Build on preprocessed tokens: combine unigrams and bigrams for title
    train_df['Tokenized_Title_UniBi'] = train_df['Tokenized_Title_Preproc'].apply(combine_unigrams_bigrams)
    test_df['Tokenized_Title_UniBi'] = test_df['Tokenized_Title_Preproc'].apply(combine_unigrams_bigrams)
    
    nb_title_unibi = NaiveBayes()
    nb_title_unibi.fit(train_df, smoothening=1.0, class_col="Class Index", text_col="Tokenized_Title_UniBi")
    train_df = nb_title_unibi.predict(train_df, text_col="Tokenized_Title_UniBi", predicted_col="Predicted_Title_UniBi")
    test_df = nb_title_unibi.predict(test_df, text_col="Tokenized_Title_UniBi", predicted_col="Predicted_Title_UniBi")
    
    print("Training Set Performance (Title Unigrams+Bigrams):")
    evaluate_model(train_df, true_col="Class Index", pred_col="Predicted_Title_UniBi")
    print("Test Set Performance (Title Unigrams+Bigrams):")
    evaluate_model(test_df, true_col="Class Index", pred_col="Predicted_Title_UniBi")

    ########
    # Experiment 6: Combining Title and Description
    ########
    print("=== Experiment 6a: Combined Unigram + Bigram Features ===")
    # Concatenate preprocessed title and description tokens, then generate unigrams + bigrams
    train_df['Tokenized_Combined_UniBi'] = train_df.apply(
        lambda row: combine_unigrams_bigrams(row['Tokenized_Title_Preproc'] + row['Tokenized Description_Preproc']), axis=1)
    test_df['Tokenized_Combined_UniBi'] = test_df.apply(
        lambda row: combine_unigrams_bigrams(row['Tokenized_Title_Preproc'] + row['Tokenized Description_Preproc']), axis=1)
    
    nb_combined_unibi = NaiveBayes()
    nb_combined_unibi.fit(train_df, smoothening=1.0, class_col="Class Index", text_col="Tokenized_Combined_UniBi")
    train_df = nb_combined_unibi.predict(train_df, text_col="Tokenized_Combined_UniBi", predicted_col="Predicted_Combined_UniBi")
    test_df = nb_combined_unibi.predict(test_df, text_col="Tokenized_Combined_UniBi", predicted_col="Predicted_Combined_UniBi")
    
    print("Training Set Performance (Combined Unigrams+Bigrams):")
    evaluate_model(train_df, true_col="Class Index", pred_col="Predicted_Combined_UniBi")
    print("Test Set Performance (Combined Unigrams+Bigrams):")
    evaluate_model(test_df, true_col="Class Index", pred_col="Predicted_Combined_UniBi")
    print("=== Experiment 6b: Separate Parameters for Title & Description ===")
    # Here we implement a separate-parameter model.
    # For clarity, we define a new class:
    class NaiveBayesSeparate:
        def __init__(self):
            self.class_priors = {}
            self.title_word_counts = {}
            self.desc_word_counts = {}
            self.total_title_words = {}
            self.total_desc_words = {}
            self.title_vocab = set()
            self.desc_vocab = set()
            self.alpha = None
            self.classes = None

        def fit(self, df, smoothening, class_col="Class Index",
                title_col="Tokenized_Title_Preproc", desc_col="Tokenized Description_Preproc"):
            self.alpha = smoothening
            self.classes = df[class_col].unique()
            self.class_priors = {}
            self.title_word_counts = {c: defaultdict(int) for c in self.classes}
            self.desc_word_counts = {c: defaultdict(int) for c in self.classes}
            self.total_title_words = {c: 0 for c in self.classes}
            self.total_desc_words = {c: 0 for c in self.classes}
            doc_counts = {c: 0 for c in self.classes}
            for _, row in df.iterrows():
                c = row[class_col]
                doc_counts[c] += 1
                for token in row[title_col]:
                    self.title_word_counts[c][token] += 1
                    self.total_title_words[c] += 1
                    self.title_vocab.add(token)
                for token in row[desc_col]:
                    self.desc_word_counts[c][token] += 1
                    self.total_desc_words[c] += 1
                    self.desc_vocab.add(token)
            total_docs = len(df)
            for c in self.classes:
                self.class_priors[c] = np.log(doc_counts[c] / total_docs)
            V_title = len(self.title_vocab)
            V_desc = len(self.desc_vocab)
            self.V_title = V_title
            self.V_desc = V_desc
            self.title_log_likelihoods = {c: {} for c in self.classes}
            self.desc_log_likelihoods = {c: {} for c in self.classes}
            for c in self.classes:
                for word in self.title_vocab:
                    count = self.title_word_counts[c].get(word, 0)
                    self.title_log_likelihoods[c][word] = np.log((count + self.alpha) / (self.total_title_words[c] + self.alpha * V_title))
                for word in self.desc_vocab:
                    count = self.desc_word_counts[c].get(word, 0)
                    self.desc_log_likelihoods[c][word] = np.log((count + self.alpha) / (self.total_desc_words[c] + self.alpha * V_desc))

        def predict(self, df, title_col="Tokenized_Title_Preproc", desc_col="Tokenized Description_Preproc", predicted_col="Predicted_Separate"):
            predictions = []
            for _, row in df.iterrows():
                class_scores = {}
                for c in self.classes:
                    score = self.class_priors[c]
                    for token in row[title_col]:
                        if token in self.title_vocab:
                            score += self.title_log_likelihoods[c][token]
                        else:
                            score += np.log(self.alpha / (self.total_title_words[c] + self.alpha * self.V_title))
                    for token in row[desc_col]:
                        if token in self.desc_vocab:
                            score += self.desc_log_likelihoods[c][token]
                        else:
                            score += np.log(self.alpha / (self.total_desc_words[c] + self.alpha * self.V_desc))
                    class_scores[c] = score
                predictions.append(max(class_scores, key=class_scores.get))
            df[predicted_col] = predictions
            return df

    nb_sep = NaiveBayesSeparate()
    nb_sep.fit(train_df, smoothening=1.0, class_col="Class Index",
               title_col="Tokenized_Title_Preproc", desc_col="Tokenized Description_Preproc")
    train_df = nb_sep.predict(train_df, title_col="Tokenized_Title_Preproc", desc_col="Tokenized Description_Preproc", predicted_col="Predicted_Separate")
    test_df = nb_sep.predict(test_df, title_col="Tokenized_Title_Preproc", desc_col="Tokenized Description_Preproc", predicted_col="Predicted_Separate")
    
    print("Training Set Performance (Separate Parameters):")
    evaluate_model(train_df, true_col="Class Index", pred_col="Predicted_Separate")
    print("Test Set Performance (Separate Parameters):")
    evaluate_model(test_df, true_col="Class Index", pred_col="Predicted_Separate")
    
    ########
    # Experiment 7: Baseline Comparisons
    ########
    print("=== Experiment 7: Baseline Comparisons ===")
    def random_baseline(df, class_col="Class Index"):
        np.random.seed(42)
        random_preds = np.random.choice(df[class_col].unique(), size=len(df))
        acc = accuracy_score(df[class_col], random_preds)
        print("Random Baseline Accuracy: {:.2f}%".format(acc * 100))
    
    def majority_baseline(df, class_col="Class Index"):
        majority_class = df[class_col].mode()[0]
        preds = [majority_class] * len(df)
        acc = accuracy_score(df[class_col], preds)
        print("Majority Baseline Accuracy: {:.2f}%".format(acc * 100))
    
    print("Validation (Test) Baselines:")
    random_baseline(test_df)
    majority_baseline(test_df)
    
    ########
    # Experiment 9: Additional Feature Engineering (POS Tags)
    ########
    # print("=== Experiment 9: Adding POS Tag Features ===")
    # # Append POS tag tokens to the preprocessed description tokens.
    # train_df['Tokenized_Desc_POS'] = train_df['Tokenized Description_Preproc'].apply(add_pos_tags)
    # test_df['Tokenized_Desc_POS'] = test_df['Tokenized Description_Preproc'].apply(add_pos_tags)
    
    # nb_pos = NaiveBayes()
    # nb_pos.fit(train_df, smoothening=1.0, class_col="Class Index", text_col="Tokenized_Desc_POS")
    # train_df = nb_pos.predict(train_df, text_col="Tokenized_Desc_POS", predicted_col="Predicted_POS")
    # test_df = nb_pos.predict(test_df, text_col="Tokenized_Desc_POS", predicted_col="Predicted_POS")
    
    # print("Training Set Performance (POS Enhanced):")
    # evaluate_model(train_df, true_col="Class Index", pred_col="Predicted_POS")
    # print("Test Set Performance (POS Enhanced):")
    # evaluate_model(test_df, true_col="Class Index", pred_col="Predicted_POS")
    
    print("=== Experiment 9: Additional Feature Engineering ===")
    # Map existing processed token columns to common names for consistency.
    train_df['Processed Description'] = train_df['Tokenized Description_Preproc']
    test_df['Processed Description'] = test_df['Tokenized Description_Preproc']
    train_df['Processed Title'] = train_df['Tokenized_Title_Preproc']
    test_df['Processed Title'] = test_df['Tokenized_Title_Preproc']

    # Calculate word probabilities from the preprocessed description model.
    # We use the nb_preproc model from Experiment 2.
    nb_model_processed = nb_preproc
    nb_model_processed.word_probs = {}
    for c in nb_model_processed.classes:
        nb_model_processed.word_probs[c] = {}
        for word in nb_model_processed.vocab:
            # Use log likelihoods to recover probabilities.
            nb_model_processed.word_probs[c][word] = nb_model_processed.log_likelihoods[c].get(
                word,
                np.log(nb_model_processed.alpha / 
                       (nb_model_processed.total_words[c] + nb_model_processed.alpha * nb_model_processed.V))
            )

    def get_discriminative_words(nb_model, num_words=10):
        """Get most discriminative words for each class."""
        discriminative_words = {c: [] for c in nb_model.classes}
        for word in nb_model.vocab:
            for c in nb_model.classes:
                # Calculate probability for the word in the current class.
                word_prob_in_class = np.exp(nb_model.word_probs[c][word])
                # Average probability for this word in the other classes.
                other_probs = [np.exp(nb_model.word_probs[other_c][word])
                               for other_c in nb_model.classes if other_c != c]
                word_prob_in_other_classes = np.mean(other_probs) if other_probs else 0
                if word_prob_in_other_classes > 0:
                    discriminative_power = word_prob_in_class / word_prob_in_other_classes
                    discriminative_words[c].append((word, discriminative_power))
        # For each class, sort the words by discriminative power and keep the top num_words.
        for c in nb_model.classes:
            discriminative_words[c] = sorted(discriminative_words[c], key=lambda x: x[1], reverse=True)[:num_words]
        return discriminative_words

    # Get discriminative words from the processed model.
    discriminative_words = get_discriminative_words(nb_model_processed, num_words=10)
    classes = nb_model_processed.classes

    # Add count features for each class based on discriminative words.
    for c in classes:
        top_words = [word for word, _ in discriminative_words[c]]
        feature_name = f'Class_{c}_KeywordCount'
        train_df[feature_name] = train_df['Processed Description'].apply(
            lambda tokens: sum(1 for token in tokens if token in top_words)
        )
        test_df[feature_name] = test_df['Processed Description'].apply(
            lambda tokens: sum(1 for token in tokens if token in top_words)
        )

    # Bucket these counts into categories: 0 as 'none', 1-2 as 'low', and 3+ as 'high'.
    for c in classes:
        feature_name = f'Class_{c}_KeywordCount'
        cat_feature = f'{feature_name}_Cat'
        train_df[cat_feature] = train_df[feature_name].apply(
            lambda x: 'none' if x == 0 else ('low' if x <= 2 else 'high')
        )
        test_df[cat_feature] = test_df[feature_name].apply(
            lambda x: 'none' if x == 0 else ('low' if x <= 2 else 'high')
        )

    # Combine all features: processed title tokens + processed description tokens +
    # the categorical discriminative keyword count features.
    def combine_features(row):
        features = row['Processed Title'] + row['Processed Description']
        for c in classes:
            cat_feature = f'Class_{c}_KeywordCount_Cat'
            features.append(f"class{c}_key_{row[cat_feature]}")
        return features

    train_df['Combined_Features'] = train_df.apply(combine_features, axis=1)
    test_df['Combined_Features'] = test_df.apply(combine_features, axis=1)

    # Train and evaluate a new Naïve Bayes model using the enhanced combined features.
    nb_model_enhanced = NaiveBayes()
    nb_model_enhanced.fit(train_df, smoothening=1.0, text_col='Combined_Features')
    train_df = nb_model_enhanced.predict(train_df, text_col='Combined_Features', predicted_col='Predicted_Enhanced')
    test_df = nb_model_enhanced.predict(test_df, text_col='Combined_Features', predicted_col='Predicted_Enhanced')
    
    print("Training Set Performance (Enhanced Features):")
    evaluate_model(train_df, true_col="Class Index", pred_col="Predicted_Enhanced")
    print("Test Set Performance (Enhanced Features):")
    evaluate_model(test_df, true_col="Class Index", pred_col="Predicted_Enhanced")
    print("=== End of Naïve Bayes' Experiments ===")