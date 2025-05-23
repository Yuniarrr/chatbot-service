import joblib


def load_model(model_path):
    """Load the trained model from a .pkl file"""
    model_data = joblib.load(model_path)
    pipeline = model_data["pipeline"]
    return pipeline


def predict_loop(model):
    print("📥 Ketik pertanyaan Anda (ketik 'exit' untuk keluar)\n")
    while True:
        question = input("❓ Pertanyaan: ").strip()
        if question.lower() in ("exit", "quit"):
            print("👋 Keluar.")
            break
        if not question:
            continue

        # Prediksi
        try:
            prediction = model.predict([question])[0]
            print(f"✅ Koleksi: {prediction}")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([question])
                if hasattr(proba, "max"):
                    confidence = proba.max()
                    print(f"📊 Confidence: {confidence:.3f}")
        except Exception as e:
            print(f"⚠️ Error saat memproses pertanyaan: {e}")
        print("-" * 50)


if __name__ == "__main__":
    MODEL_PATH = "./notebooks/predict_classification/best_improved_question_classifier.pkl"  # ganti jika perlu
    print(f"📦 Loading model from {MODEL_PATH} ...")
    model = load_model(MODEL_PATH)
    predict_loop(model)
