import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pickle
import re
import string
from collections import Counter
import joblib
import time
import warnings
from sklearn.metrics import f1_score
from setfit import SetFitModel, SetFitTrainer
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import T5ForConditionalGeneration, T5Tokenizer

warnings.filterwarnings("ignore")


class TextAugmentor:
    def __init__(self, model_name="t5-small"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def paraphrase(self, text, num_return_sequences=3):
        input_text = "paraphrase: " + text + " </s>"
        encoding = self.tokenizer.encode_plus(
            input_text, padding="longest", return_tensors="pt"
        )
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=256,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=num_return_sequences,
        )

        paraphrased_texts = [
            self.tokenizer.decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for output in outputs
        ]
        return paraphrased_texts


class SetFitClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        num_iterations=20,
        batch_size=16,
    ):
        self.model_id = model_id
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.model = None
        self.trainer = None

    def fit(self, X, y):
        # Load pretrained SetFit model
        self.model = SetFitModel.from_pretrained(self.model_id)

        # Siapkan trainer dengan parameter batch_size dan eval_every
        self.trainer = SetFitTrainer(
            model=self.model,
            train_dataset={"text": X.tolist(), "label": y.tolist()},
            batch_size=self.batch_size,
        )

        # Training dengan num_iterations yang bisa diatur
        self.trainer.train(num_iterations=self.num_iterations)
        return self

    def predict(self, X):
        return self.model(X.tolist())

    def predict_proba(self, X):
        return self.model.predict_proba(X.tolist())


class ImprovedQuestionClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.pipeline = None
        self.label_encoder = None
        self.class_distribution = None
        self.best_params = None

    def advanced_preprocess_text(self, text):
        """Advanced preprocessing for Indonesian text"""
        if pd.isna(text) or text == "":
            return ""

        # Convert to lowercase
        text = str(text).lower()

        # Remove URLs, emails, and special patterns
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )
        text = re.sub(r"\S+@\S+", "", text)

        # Normalize common Indonesian question words and patterns
        indonesian_normalizations = {
            r"\bapa\s+itu\b": "apa_itu",
            r"\bbagaimana\s+cara\b": "bagaimana_cara",
            r"\bsiapa\s+yang\b": "siapa_yang",
            r"\bkapan\s+waktu\b": "kapan_waktu",
            r"\bdimana\s+tempat\b": "dimana_tempat",
            r"\bmengapa\s+kenapa\b": "mengapa_kenapa",
            r"\bberapa\s+jumlah\b": "berapa_jumlah",
        }

        for pattern, replacement in indonesian_normalizations.items():
            text = re.sub(pattern, replacement, text)

        # Remove excessive punctuation but keep question marks
        text = re.sub(r"[^\w\s\?]", " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def analyze_data_quality(self, df):
        """Analyze data quality and class distribution"""
        print("=" * 60)
        print("DATA QUALITY ANALYSIS")
        print("=" * 60)

        # Basic statistics
        print(f"Total samples: {len(df)}")
        print(f"Unique collections: {df['collection_name'].nunique()}")
        print()

        # Class distribution
        class_dist = df["collection_name"].value_counts()
        print("Class Distribution:")
        for class_name, count in class_dist.items():
            percentage = (count / len(df)) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")

        # Identify imbalanced classes
        min_samples = class_dist.min()
        max_samples = class_dist.max()
        imbalance_ratio = max_samples / min_samples

        print(f"\nImbalance Ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio > 3:
            print("⚠️  HIGH CLASS IMBALANCE DETECTED!")
            print("   Consider using SMOTE or class weighting")

        # Check for very small classes
        small_classes = class_dist[class_dist < 5]
        if len(small_classes) > 0:
            print(f"\n⚠️  Classes with < 5 samples: {list(small_classes.index)}")
            print("   These classes may cause overfitting")

        # Text length analysis
        df["text_length"] = df["processed_question"].str.len()
        df["word_count"] = df["processed_question"].str.split().str.len()

        print(f"\nText Statistics:")
        print(f"  Average text length: {df['text_length'].mean():.1f} characters")
        print(f"  Average word count: {df['word_count'].mean():.1f} words")
        print(f"  Min/Max words: {df['word_count'].min()}/{df['word_count'].max()}")

        return class_dist

    def load_data(self, json_file_path):
        """Load and clean data from JSON file"""
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Clean data
        print("Cleaning data...")
        initial_count = len(df)

        # Remove entries with null/empty collection_name or question
        df = df.dropna(subset=["collection_name", "question"])
        df = df[df["collection_name"].str.strip() != ""]
        df = df[df["question"].str.strip() != ""]

        # Remove duplicates
        df = df.drop_duplicates(subset=["question", "collection_name"])

        print(f"Removed {initial_count - len(df)} invalid/duplicate entries")

        # Advanced preprocessing
        df["processed_question"] = df["question"].apply(self.advanced_preprocess_text)

        # Remove entries where preprocessing resulted in empty strings
        df = df[df["processed_question"].str.len() > 0]

        return df

    def prepare_data_from_list(self, data_list):
        """Prepare data from list of dictionaries with enhanced cleaning"""
        df = pd.DataFrame(data_list)

        # Clean data
        print("Cleaning data...")
        initial_count = len(df)

        # Remove invalid entries
        df = df.dropna(subset=["collection_name", "question"])
        df = df[df["collection_name"].astype(str).str.strip() != ""]
        df = df[df["question"].astype(str).str.strip() != ""]

        # Remove duplicates
        df = df.drop_duplicates(subset=["question", "collection_name"])

        print(f"Removed {initial_count - len(df)} invalid/duplicate entries")

        # Advanced preprocessing
        df["processed_question"] = df["question"].apply(self.advanced_preprocess_text)

        # Remove entries where preprocessing resulted in empty strings
        df = df[df["processed_question"].str.len() > 0]
        df = df.reset_index(drop=True)

        return df

    def get_optimized_vectorizer(self, df, custom_params=None):
        """Get optimized TF-IDF vectorizer based on data characteristics"""
        total_samples = len(df)
        unique_classes = df["collection_name"].nunique()

        if custom_params:
            # Use custom parameters from hyperparameter tuning
            return TfidfVectorizer(
                max_features=custom_params.get("max_features", 5000),
                ngram_range=custom_params.get("ngram_range", (1, 2)),
                min_df=custom_params.get("min_df", 1),
                max_df=custom_params.get("max_df", 0.95),
                stop_words=None,
                sublinear_tf=True,
                use_idf=True,
                smooth_idf=True,
                norm="l2",
            )

        # Default adaptive parameters
        if total_samples < 1000:
            max_features = min(2000, total_samples * 2)
            ngram_range = (1, 2)
        elif total_samples < 5000:
            max_features = 5000
            ngram_range = (1, 3)
        else:
            max_features = 10000
            ngram_range = (1, 3)

        # Calculate optimal min_df
        min_df = max(1, int(total_samples * 0.001))
        max_df = 0.95

        print(f"Vectorizer settings:")
        print(f"  Max features: {max_features}")
        print(f"  N-gram range: {ngram_range}")
        print(f"  Min/Max DF: {min_df}/{max_df}")

        return TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words=None,
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True,
            norm="l2",
        )

    def smart_hyperparameter_tuning(self, df, quick_tune=True):
        """Improved hyperparameter tuning with better scoring and search strategy"""
        print("\n" + "=" * 60)
        print("SMART HYPERPARAMETER TUNING (F1-macro + Randomized)")
        print("=" * 60)

        X = df["processed_question"]
        y = df["collection_name"]

        # Baseline setup
        baseline_vectorizer = self.get_optimized_vectorizer(df)
        baseline_model = LogisticRegression(
            random_state=42, max_iter=2000, class_weight="balanced", solver="liblinear"
        )
        baseline_pipeline = Pipeline(
            [("vectorizer", baseline_vectorizer), ("classifier", baseline_model)]
        )

        # Evaluate baseline with F1-macro
        class_dist = y.value_counts()
        min_class_count = class_dist.min()

        try:
            cv_folds = min(5, min_class_count) if min_class_count >= 3 else 2
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            baseline_scores = cross_val_score(
                baseline_pipeline, X, y, cv=cv, scoring="f1_macro"
            )
            baseline_mean = baseline_scores.mean()
            baseline_std = baseline_scores.std()
            print(f"Baseline F1-macro: {baseline_mean:.4f} ± {baseline_std:.4f}")
        except Exception as e:
            print(f"Baseline evaluation failed: {e}")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            baseline_pipeline.fit(X_train, y_train)
            y_pred = baseline_pipeline.predict(X_test)
            baseline_mean = f1_score(y_test, y_pred, average="macro")
            print(f"Baseline F1-macro (train-test): {baseline_mean:.4f}")

        # Hyperparameter space
        param_grid = {
            "vectorizer__max_features": [1000, 2000, 5000],
            "vectorizer__ngram_range": [(1, 1), (1, 2)],
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__class_weight": [None, "balanced"],  # ← tambahkan ini
        }

        cv_folds = min(3, min_class_count)
        print(
            f"Using {'RandomizedSearchCV' if quick_tune else 'GridSearchCV'} with {cv_folds} folds"
        )

        tuning_pipeline = Pipeline(
            [
                ("vectorizer", TfidfVectorizer(stop_words=None, sublinear_tf=True)),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=42,
                        max_iter=2000,
                        class_weight="balanced",
                        solver="liblinear",
                    ),
                ),
            ]
        )

        search_class = RandomizedSearchCV if quick_tune else GridSearchCV
        search_kwargs = {
            "param_distributions" if quick_tune else "param_grid": param_grid,
            "cv": StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            "scoring": "f1_macro",
            "n_jobs": -1,
            "verbose": 1,
            "error_score": "raise",
            "n_iter": 15 if quick_tune else None,  # Only for RandomizedSearchCV
        }

        print("Starting hyperparameter search...")
        start_time = time.time()
        search = search_class(tuning_pipeline, **search_kwargs)
        search.fit(X, y)
        tuning_time = time.time() - start_time

        print(f"Search completed in {tuning_time:.2f}s")
        print(f"Best parameters: {search.best_params_}")
        print(f"Best F1-macro CV score: {search.best_score_:.4f}")

        improvement = search.best_score_ - baseline_mean
        print(f"Improvement over baseline: {improvement:+.4f}")

        if improvement > 0.01:
            print("✅ Using tuned model")
            self.best_params = search.best_params_
            self.pipeline = search.best_estimator_
            return search.best_score_
        else:
            print("⚠️  No significant improvement - using baseline")
            self.pipeline = baseline_pipeline
            self.pipeline.fit(X, y)
            return baseline_mean

    def train_model(
        self,
        df,
        use_class_balancing=True,
        use_ensemble=True,
        min_samples_threshold=5,
        enable_tuning=False,
    ):
        """Train optimized classification model with optional hyperparameter tuning"""

        # Analyze data quality first
        class_dist = self.analyze_data_quality(df)

        # If hyperparameter tuning is enabled, do it first
        if enable_tuning:
            tuned_score = self.smart_hyperparameter_tuning(df, quick_tune=True)
            if self.pipeline is not None:
                print(f"Using tuned model with score: {tuned_score:.4f}")
                return tuned_score

        X = df["processed_question"]
        y = df["collection_name"]

        # Check if we have enough data for reliable train/test split
        min_class_count = class_dist.min()
        if min_class_count < 2:
            print("⚠️  Some classes have only 1 sample! Consider collecting more data.")
            return 0.0

        # Filter out classes with too few samples if requested
        if min_samples_threshold > 2:
            classes_to_keep = class_dist[class_dist >= min_samples_threshold].index
            if len(classes_to_keep) < len(class_dist):
                print(
                    f"⚠️  Filtering out {len(class_dist) - len(classes_to_keep)} classes with < {min_samples_threshold} samples"
                )
                df_filtered = df[df["collection_name"].isin(classes_to_keep)].copy()
                X = df_filtered["processed_question"]
                y = df_filtered["collection_name"]
                class_dist = y.value_counts()
                min_class_count = class_dist.min()

        # Stratified split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            print("⚠️  Cannot stratify - some classes may have too few samples")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        # Get optimized vectorizer (use tuned parameters if available)
        vectorizer_params = None
        if self.best_params:
            vectorizer_params = {
                "max_features": self.best_params.get("vectorizer__max_features"),
                "ngram_range": self.best_params.get("vectorizer__ngram_range"),
                "min_df": self.best_params.get("vectorizer__min_df", 1),
            }

        vectorizer = self.get_optimized_vectorizer(df, vectorizer_params)

        # Prepare models with optimized parameters
        models = {}

        # Naive Bayes - good for text classification
        models["naive_bayes"] = MultinomialNB(alpha=0.01)

        # Logistic Regression with class balancing
        # lr_params = {"random_state": 42, "max_iter": 2000, "solver": "liblinear"}
        # if self.best_params and "classifier__C" in self.best_params:
        #     lr_params["C"] = self.best_params["classifier__C"]
        # else:
        #     lr_params["C"] = 1.0

        # if use_class_balancing:
        #     lr_params["class_weight"] = "balanced"
        class_weights = None
        if use_class_balancing:
            unique_classes = np.unique(y)
            class_weights_array = compute_class_weight(
                class_weight="balanced", classes=unique_classes, y=y
            )
            class_weights = dict(zip(unique_classes, class_weights_array))
            print(f"Computed class weights: {class_weights}")

        lr_params = {
            "random_state": 42,
            "max_iter": 2000,
            "solver": "liblinear",
            "C": (
                self.best_params.get("classifier__C", 1.0) if self.best_params else 1.0
            ),
        }

        if use_class_balancing:
            lr_params["class_weight"] = class_weights

        models["logistic_regression"] = LogisticRegression(**lr_params)
        models["setfit"] = SetFitClassifier(num_iterations=100, batch_size=40)

        # SVM with balanced classes (only for smaller datasets)
        if len(df) < 10000:
            svm_params = {
                "kernel": "linear",
                "random_state": 42,
                "probability": True,
                "C": 1.0,
            }
            if use_class_balancing:
                # svm_params["class_weight"] = "balanced"
                svm_params["class_weight"] = class_weights
            models["svm"] = SVC(**svm_params)

        # Random Forest
        rf_params = {
            "n_estimators": 100,
            "random_state": 42,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "n_jobs": -1,
        }
        if use_class_balancing:
            rf_params["class_weight"] = "balanced"
        models["random_forest"] = RandomForestClassifier(**rf_params)

        best_model = None
        best_accuracy = 0
        best_model_name = ""

        print("\n" + "=" * 60)
        print("MODEL TRAINING AND EVALUATION")
        print("=" * 60)

        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name.upper()}...")

            try:
                # Create pipeline with safe SMOTE configuration
                use_smote = (
                    use_class_balancing
                    and min_class_count >= 3
                    and name in ["logistic_regression", "svm"]
                    and min_class_count < 20
                )

                if use_smote:
                    # Calculate safe k_neighbors for SMOTE
                    safe_k_neighbors = min(3, min_class_count - 1)

                    if safe_k_neighbors >= 1:
                        print(f"  Using SMOTE with k_neighbors={safe_k_neighbors}")
                        try:
                            pipeline = ImbPipeline(
                                [
                                    ("vectorizer", vectorizer),
                                    (
                                        "smote",
                                        SMOTE(
                                            random_state=42,
                                            k_neighbors=safe_k_neighbors,
                                        ),
                                    ),
                                    ("classifier", model),
                                ]
                            )
                        except Exception as smote_error:
                            print(f"  SMOTE failed: {smote_error}")
                            print(
                                "  Falling back to regular pipeline with class weights"
                            )
                            if name == "setfit":
                                pipeline = Pipeline([("classifier", model)])
                            else:
                                pipeline = Pipeline(
                                    [("vectorizer", vectorizer), ("classifier", model)]
                                )
                            # pipeline = Pipeline(
                            #     [("vectorizer", vectorizer), ("classifier", model)]
                            # )
                    else:
                        pipeline = Pipeline(
                            [("vectorizer", vectorizer), ("classifier", model)]
                        )
                else:
                    pipeline = Pipeline(
                        [("vectorizer", vectorizer), ("classifier", model)]
                    )

                # Train with error handling
                start_time = time.time()

                # Use simple train-test evaluation for small datasets
                if min_class_count < 5:
                    print("  Using simple train-test split due to small class sizes")
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    cv_mean = accuracy
                    cv_std = 0.0
                else:
                    # Use cross-validation for larger datasets
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)

                    # Cross-validation with error handling
                    try:
                        cv_folds = min(3, min_class_count)
                        cv = StratifiedKFold(
                            n_splits=cv_folds, shuffle=True, random_state=42
                        )
                        cv_scores = cross_val_score(
                            pipeline, X_train, y_train, cv=cv, scoring="accuracy"
                        )
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()
                    except Exception as cv_error:
                        print(f"  Cross-validation failed: {cv_error}")
                        cv_mean = accuracy
                        cv_std = 0.0

                training_time = time.time() - start_time

                print(f"  Training time: {training_time:.2f}s")
                print(f"  Test accuracy: {accuracy:.4f}")
                print(f"  CV accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

                # Update best model
                score_to_compare = cv_mean if cv_mean > 0 else accuracy
                if score_to_compare > best_accuracy:
                    best_accuracy = score_to_compare
                    best_model = pipeline
                    best_model_name = name

            except Exception as e:
                print(f"  Error training {name}: {e}")
                continue

        if best_model is None:
            print("⚠️  No model trained successfully!")
            return 0.0

        # Create ensemble if conditions are met
        if use_ensemble and len(models) >= 2 and min_class_count >= 5:
            print(f"\nCreating ensemble model...")
            try:
                working_models = []
                for name, model in models.items():
                    if name != best_model_name:
                        try:
                            pipeline = Pipeline(
                                [("vectorizer", vectorizer), ("classifier", model)]
                            )
                            pipeline.fit(X_train, y_train)
                            working_models.append((name, pipeline))
                        except:
                            continue

                if working_models and len(working_models) >= 1:
                    working_models.append((best_model_name, best_model))

                    voting_classifier = VotingClassifier(
                        estimators=working_models[:3], voting="soft"
                    )

                    voting_classifier.fit(X_train, y_train)
                    ensemble_pred = voting_classifier.predict(X_test)
                    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

                    print(f"  Ensemble accuracy: {ensemble_accuracy:.4f}")

                    if ensemble_accuracy > best_accuracy:
                        best_model = voting_classifier
                        best_accuracy = ensemble_accuracy
                        best_model_name = "ensemble"

            except Exception as e:
                print(f"  Ensemble creation failed: {e}")

        # Final evaluation
        self.pipeline = best_model

        print(f"\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Best model: {best_model_name}")
        print(f"Best accuracy: {best_accuracy:.4f}")

        # Detailed classification report
        y_pred_final = self.pipeline.predict(X_test)
        final_accuracy = accuracy_score(y_test, y_pred_final)

        print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred_final))

        # Prediction speed test
        test_questions = X_test.iloc[: min(10, len(X_test))].tolist()
        start_time = time.time()
        _ = self.pipeline.predict(test_questions)
        prediction_time = time.time() - start_time
        print(
            f"\nPrediction speed: {prediction_time/len(test_questions):.4f}s per question"
        )

        return final_accuracy

    def predict(self, question):
        """Predict collection_name for a single question"""
        if self.pipeline is None:
            raise ValueError("Model not trained yet!")

        processed_question = self.advanced_preprocess_text(question)
        if not processed_question:
            return "unknown", 0.0

        prediction = self.pipeline.predict([processed_question])[0]

        try:
            probabilities = self.pipeline.predict_proba([processed_question])[0]
            confidence = max(probabilities)
        except:
            confidence = 0.5

        return prediction, confidence

    def predict_batch(self, questions):
        """Predict collection_name for multiple questions"""
        if self.pipeline is None:
            raise ValueError("Model not trained yet!")

        processed_questions = [self.advanced_preprocess_text(q) for q in questions]
        predictions = self.pipeline.predict(processed_questions)

        try:
            probabilities = self.pipeline.predict_proba(processed_questions)
            confidences = [max(prob) for prob in probabilities]
        except:
            confidences = [0.5] * len(predictions)

        return list(zip(predictions, confidences))

    def save_model(self, filepath):
        """Save the trained model"""
        if self.pipeline is None:
            raise ValueError("No model to save!")

        # Save both the pipeline and the best parameters
        model_data = {"pipeline": self.pipeline, "best_params": self.best_params}
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        if isinstance(model_data, dict):
            self.pipeline = model_data["pipeline"]
            self.best_params = model_data.get("best_params")
        else:
            # Backward compatibility
            self.pipeline = model_data
        print(f"Model loaded from {filepath}")


# Utility functions for data improvement
def suggest_data_improvements(df):
    """Suggest improvements for better accuracy"""
    print("\n" + "=" * 60)
    print("DATA IMPROVEMENT SUGGESTIONS")
    print("=" * 60)

    class_dist = df["collection_name"].value_counts()
    suggestions = []

    # Check class balance
    min_samples = class_dist.min()
    max_samples = class_dist.max()
    imbalance_ratio = max_samples / min_samples

    if imbalance_ratio > 5:
        suggestions.append(
            f"🔄 Balance your classes - ratio is {imbalance_ratio:.1f}:1"
        )
        for class_name, count in class_dist.items():
            if count < min_samples * 2:
                suggestions.append(
                    f"   • Add more samples for '{class_name}' (currently {count})"
                )

    # Check minimum samples per class
    if min_samples < 10:
        suggestions.append(f"📈 Increase minimum samples per class to at least 10")

    # Check for duplicate or similar questions
    questions_per_class = df.groupby("collection_name")["question"].apply(list)
    for class_name, questions in questions_per_class.items():
        if len(set(questions)) != len(questions):
            suggestions.append(f"🔍 Remove duplicate questions in '{class_name}' class")

    # Check text quality
    avg_length = df["processed_question"].str.len().mean()
    if avg_length < 20:
        suggestions.append(
            f"📝 Improve question quality - average length is only {avg_length:.1f} chars"
        )

    if suggestions:
        for suggestion in suggestions:
            print(suggestion)
        print(
            f"\n💡 Target: Aim for 20-50+ samples per class with balanced distribution"
        )
    else:
        print("✅ Your data looks good for training!")

    return suggestions


def augment_minority_class(df, augmentor, imbalance_threshold=5, min_samples=10):
    class_counts = df["collection_name"].value_counts()
    max_count = class_counts.max()
    augmented_rows = []

    for cls, count in class_counts.items():
        imbalance_ratio = max_count / count
        if imbalance_ratio > imbalance_threshold and count < min_samples:
            print(
                f"Augmenting class '{cls}' with {count} samples (ratio={imbalance_ratio:.1f})"
            )
            samples = df[df["collection_name"] == cls]["question"].tolist()
            for sample in samples:
                paraphrases = augmentor.paraphrase(sample, num_return_sequences=3)
                for p in paraphrases:
                    augmented_rows.append(
                        {
                            "question": p,
                            "collection_name": cls,
                            "processed_question": p.lower(),  # contoh proses sederhana
                        }
                    )

    df_augmented = df._append(augmented_rows, ignore_index=True)
    return df_augmented


# Example usage with improvements
def train_improved_classifier():
    """Example of training with improved hyperparameter tuning"""

    # Load your data
    with open("./notebooks/question.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    sample_data = [
        {
            "id": item["id"],
            "question": item["question"],
            "collection_name": item["collection_name"],
        }
        for item in data
        if item.get("collection_name") and item.get("question")
    ]

    classifier = ImprovedQuestionClassifier()
    df = classifier.prepare_data_from_list(sample_data)

    augmentor = TextAugmentor(model_name="t5-small")
    df = augment_minority_class(df, augmentor, imbalance_threshold=5, min_samples=10)

    suggest_data_improvements(df)

    print(f"\nTraining with {len(df)} samples...")
    accuracy = classifier.train_model(
        df,
        use_class_balancing=True,
        use_ensemble=True,
        min_samples_threshold=3,
        enable_tuning=True,
    )

    print(f"\nFinal model accuracy: {accuracy:.4f}")

    classifier.save_model("./notebooks/test/improved_question_classifier.pkl")

    return classifier, accuracy


if __name__ == "__main__":
    train_improved_classifier()
