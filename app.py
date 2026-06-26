from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "text.csv"
DISTILBERT_MODEL_PATH = BASE_DIR / "models" / "distilbert_emotion"
NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"
NLLB_TARGET_LANGUAGE = "eng_Latn"
NLLB_LANGUAGE_CODES = {
    "Chinese": "zho_Hans",
    "Malay": "zsm_Latn",
    "Tamil": "tam_Taml",
    "Indonesian": "ind_Latn",
}
MODEL_COMPARISON_PATH = BASE_DIR / "results" / "model_comparison.png"

LABEL_MAP = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise",
}

EMOTION_COLORS = {
    "Sadness": "#4c78a8",
    "Joy": "#59a14f",
    "Love": "#e15759",
    "Anger": "#b07aa1",
    "Fear": "#f28e2b",
    "Surprise": "#edc948",
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "i", "if", "in",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "she",
    "so", "that", "the", "their", "them", "then", "there", "they", "this",
    "to", "was", "we", "were", "what", "when", "where", "which", "who",
    "will", "with", "you", "your",
}

_PREPROCESSING_TOOLS = None


st.set_page_config(
    page_title="EmotionSense",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #f7fbff 0%, #ffffff 34%, #ffffff 100%);
    }
    [data-testid="stSidebar"] {
        background: #eef6ff;
        border-right: 1px solid #d7e7f7;
    }
    h1, h2, h3 {
        color: #1f2937;
        letter-spacing: 0;
    }
    h1 {
        font-weight: 800;
    }
    h2, h3 {
        margin-top: 1.2rem;
    }
    .small-note {
        color: #64748b;
        font-size: 0.92rem;
    }
    .hero-panel {
        background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
        border: 1px solid #d9e5f2;
        border-radius: 8px;
        padding: 24px 26px;
        margin: 16px 0 18px 0;
        box-shadow: 0 10px 26px rgba(76, 120, 168, 0.10);
    }
    .emotion-strip {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        height: 8px;
        overflow: hidden;
        border-radius: 8px;
        margin-bottom: 18px;
    }
    .hero-title {
        font-size: 1.35rem;
        font-weight: 750;
        margin-bottom: 8px;
        color: #1f2937;
    }
    .hero-copy {
        color: #334155;
        font-size: 1.02rem;
        line-height: 1.65;
        margin: 0;
    }
    .emotion-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 18px;
    }
    .emotion-chip {
        border-radius: 999px;
        padding: 6px 12px;
        color: #ffffff;
        font-size: 0.86rem;
        font-weight: 650;
    }
    .feature-card {
        background: #ffffff;
        border: 1px solid #d9e5f2;
        border-radius: 8px;
        padding: 16px 18px;
        min-height: 130px;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
    }
    .feature-number {
        display: inline-block;
        color: #ffffff;
        background: #4c78a8;
        border-radius: 999px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .feature-title {
        color: #1f2937;
        font-weight: 750;
        margin-bottom: 6px;
    }
    .feature-copy {
        color: #64748b;
        font-size: 0.92rem;
        line-height: 1.55;
        margin: 0;
    }
    .insight-card {
        background: #ffffff;
        border: 1px solid #d9e5f2;
        border-left: 4px solid #4c78a8;
        border-radius: 8px;
        padding: 12px 15px;
        margin: 8px 0 22px 0;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
    }
    .insight-label {
        color: #4c78a8;
        font-size: 0.78rem;
        font-weight: 750;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .insight-copy {
        color: #334155;
        font-size: 0.94rem;
        line-height: 1.55;
        margin: 0;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d9e5f2;
        border-left: 4px solid #4c78a8;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
    }
    .stButton > button {
        background: #e15759;
        border: 1px solid #e15759;
        color: #ffffff;
        border-radius: 8px;
        font-weight: 700;
        padding: 0.55rem 1rem;
    }
    .stButton > button:hover {
        background: #c94648;
        border-color: #c94648;
        color: #ffffff;
    }
    [data-testid="stDataFrame"], table {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_preprocessing_tools():
    global _PREPROCESSING_TOOLS
    if _PREPROCESSING_TOOLS is not None:
        return _PREPROCESSING_TOOLS

    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer

        try:
            stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            stop_words = set(stopwords.words("english"))

        stemmer = PorterStemmer()
        _PREPROCESSING_TOOLS = (stemmer.stem, stop_words)
    except Exception:
        _PREPROCESSING_TOOLS = (lambda word: word, STOPWORDS)

    return _PREPROCESSING_TOOLS


def clean_text(text: str) -> str:
    stem_word, stop_words = get_preprocessing_tools()
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d+", "", text)
    text = text.lower()
    words = text.split()
    words = [stem_word(word) for word in words if word not in stop_words]
    text = " ".join(words)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(int)
    df["emotion"] = df["label"].map(LABEL_MAP)
    df["clean_text"] = df["text"].astype(str).map(clean_text)
    df["word_count"] = df["clean_text"].str.split().str.len()
    df["char_count"] = df["text"].astype(str).str.len()
    return df


@st.cache_resource(show_spinner=False)
def load_translation_bundle():
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except Exception as exc:
        return None, f"Transformers translation classes are not available: {exc}"

    try:
        tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)
        model.eval()
        return {
            "model": model,
            "tokenizer": tokenizer,
            "torch": torch,
            "name": NLLB_MODEL_NAME,
        }, None
    except Exception as exc:
        return None, f"Pre-trained translation model could not be loaded: {exc}"


@st.cache_resource(show_spinner=False)
def load_distilbert_bundle():
    try:
        from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast
    except Exception as exc:
        return None, f"Transformers/TensorFlow DistilBERT is not available: {exc}"

    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(str(DISTILBERT_MODEL_PATH), local_files_only=True)
        model = TFDistilBertForSequenceClassification.from_pretrained(str(DISTILBERT_MODEL_PATH), local_files_only=True)
        return {
            "model": model,
            "tokenizer": tokenizer,
            "name": "DistilBERT",
        }, None
    except Exception as exc:
        return None, f"DistilBERT model could not be loaded: {exc}"


def maybe_translate_to_english(text: str, source_language: str) -> tuple[str, str | None]:
    if source_language == "English":
        return text, None

    source_code = NLLB_LANGUAGE_CODES.get(source_language)
    if source_code is None:
        return text, f"Unsupported translation language selected: {source_language}"

    bundle, error = load_translation_bundle()
    if error:
        return text, f"Pre-trained translation model unavailable, using original text instead. Details: {error}"

    try:
        tokenizer = bundle["tokenizer"]
        model = bundle["model"]
        torch = bundle["torch"]
        tokenizer.src_lang = source_code
        inputs = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        target_token_id = tokenizer.convert_tokens_to_ids(NLLB_TARGET_LANGUAGE)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=target_token_id,
                max_length=256,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=1.1,
                early_stopping=True,
            )
        translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return translated, None
    except Exception as exc:
        return text, f"Pre-trained translation failed, using original text instead. Details: {exc}"


def predict_with_distilbert(text: str):
    bundle, error = load_distilbert_bundle()
    if bundle is None:
        return None, error

    cleaned = clean_text(text)
    encoded = bundle["tokenizer"](
        cleaned,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=128,
    )
    outputs = bundle["model"](encoded, training=False)
    logits = outputs.logits.numpy()[0]
    probabilities = softmax(logits)
    label_index = int(np.argmax(probabilities))

    return {
        "model_name": bundle["name"],
        "cleaned_text": cleaned,
        "label_index": label_index,
        "emotion": LABEL_MAP[label_index],
        "confidence": float(probabilities[label_index]),
        "probabilities": probabilities_to_frame(probabilities),
        "influential_terms": estimate_distilbert_token_influence(bundle, cleaned, label_index, probabilities[label_index]),
    }, None


def softmax(values):
    values = np.asarray(values, dtype=float)
    exp_values = np.exp(values - np.max(values))
    return exp_values / exp_values.sum()


def estimate_distilbert_token_influence(bundle, cleaned_text: str, label_index: int, base_score: float) -> pd.DataFrame:
    words = cleaned_text.split()
    if not words:
        return pd.DataFrame()

    rows = []
    for index, word in enumerate(words[:20]):
        altered_words = words.copy()
        altered_words.pop(index)
        if not altered_words:
            continue
        altered_text = " ".join(altered_words)
        encoded = bundle["tokenizer"](
            altered_text,
            return_tensors="tf",
            truncation=True,
            padding=True,
            max_length=128,
        )
        outputs = bundle["model"](encoded, training=False)
        probabilities = softmax(outputs.logits.numpy()[0])
        score_drop = float(base_score - probabilities[label_index])
        rows.append(
            {
                "Term": word,
                "Estimated influence": score_drop,
                "Method": "Remove word and compare DistilBERT confidence",
            }
        )

    terms = pd.DataFrame(rows)
    if terms.empty:
        return terms
    terms["Absolute influence"] = terms["Estimated influence"].abs()
    return terms.sort_values("Absolute influence", ascending=False).head(10).drop(columns=["Absolute influence"])


def probabilities_to_frame(probabilities, classes=None) -> pd.DataFrame:
    if probabilities is None:
        return pd.DataFrame()
    if classes is None:
        classes = list(LABEL_MAP.keys())
    rows = []
    for raw_label, probability in zip(classes, probabilities):
        label = int(raw_label)
        rows.append({"Emotion": LABEL_MAP[label], "Probability": float(probability)})
    return pd.DataFrame(rows).sort_values("Probability", ascending=False)


def render_header(title: str, subtitle: str):
    st.title(title)
    st.caption(subtitle)


def render_emotion_badges():
    chips = "".join(
        f'<span class="emotion-chip" style="background:{color};">{emotion}</span>'
        for emotion, color in EMOTION_COLORS.items()
    )
    st.markdown(f'<div class="emotion-chip-row">{chips}</div>', unsafe_allow_html=True)


def home_page():
    render_header("EmotionSense", "Multi-class emotion detection for social media text")

    strip = "".join(f'<div style="background:{color};"></div>' for color in EMOTION_COLORS.values())
    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="emotion-strip">{strip}</div>
            <div class="hero-title">Emotion detection for short social media text</div>
            <p class="hero-copy">
                EmotionSense predicts the emotion expressed in a short text and
                classifies it into six emotion classes. The app preprocesses user
                input, converts it into model-ready features, and returns an
                emotion prediction with confidence scores.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_emotion_badges()

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset", "416,809 samples")
    col2.metric("Classes", "6 emotions")
    col3.metric("Main model", "DistilBERT")

    st.subheader("Project Workflow")
    workflow_cols = st.columns(3)
    workflow_cards = [
        ("01", "Enter Text", "Users type or paste a short text, with pre-trained model translation support for non-English input."),
        ("02", "Predict Emotion", "The app uses the selected NLP model to classify the text into one of six emotion labels."),
        ("03", "Review Output", "The result page shows confidence, class probabilities, and influential words for interpretation."),
    ]
    for column, (number, title, copy) in zip(workflow_cols, workflow_cards):
        with column:
            st.markdown(
                f"""
                <div class="feature-card">
                    <span class="feature-number">{number}</span>
                    <div class="feature-title">{title}</div>
                    <p class="feature-copy">{copy}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("How To Use")
    st.markdown(
        """
        1. Open the Text Analyzer page.
        2. Enter English text, or select another language so the pre-trained translation model can translate it to English.
        3. Click Analyze Emotion.
        4. Review the prediction, confidence score, class probabilities, and influential terms.
        """
    )

    st.subheader("Team Allocation")
    st.table(
        pd.DataFrame(
            [
                ["Wan Nur Hajar", "Text Processing & NLP 1", "Theme selection, dataset collection, preprocessing"],
                ["Chek Chee Him", "Text Processing & NLP 2", "Feature extraction, model training, evaluation"],
                ["Ng Pi Zhen", "Web Application", "Streamlit 5-page app, prediction workflow, deployment support"],
                ["Amirah", "Data Visualization", "Five visualizations and A1 poster"],
            ],
            columns=["Name", "Role", "Responsibility"],
        )
    )


def text_analyzer_page():
    render_header("Text Analyzer", "Analyze text and predict its emotion.")

    examples = {
        "Joy": "I feel so happy and excited about this amazing day.",
        "Love": "I love the way you always make me feel cared for.",
        "Anger": "I am so annoyed and angry that this keeps happening.",
        "Fear": "I feel scared and nervous about what might happen next.",
        "Sadness": "I feel lonely and heartbroken after everything that happened.",
        "Surprise": "I am amazed and shocked by this unexpected result.",
    }

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_example = st.selectbox("Example text", list(examples.keys()))
    with col2:
        source_language = st.selectbox(
            "Input language",
            ["English", "Chinese", "Malay", "Tamil", "Indonesian"],
        )

    user_text = st.text_area(
        "Input text",
        value=examples[selected_example],
        height=150,
        placeholder="Type or paste a social media text here...",
    )

    use_translation = source_language != "English"
    st.caption("Emotion model: DistilBERT")
    if use_translation:
        st.caption(f"Translation model: {NLLB_MODEL_NAME} translates {source_language} input to English before prediction.")

    if st.button("Analyze Emotion", type="primary"):
        if not user_text.strip():
            st.warning("Please enter text before analyzing.")
            return

        translated_text = user_text
        translation_warning = None
        if use_translation:
            translated_text, translation_warning = maybe_translate_to_english(user_text, source_language)

        if translation_warning:
            st.warning(translation_warning)

        if use_translation:
            st.subheader("Translated English Text")
            st.write(translated_text)

        result, error = predict_with_distilbert(translated_text)
        if error:
            st.error(f"DistilBERT model could not be loaded. Details: {error}")
            return

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted emotion", result["emotion"])
        col2.metric("Confidence", f"{result['confidence']:.1%}" if result["confidence"] is not None else "N/A")
        col3.metric("Model used", result["model_name"])

        st.subheader("Preprocessed Text")
        st.code(result["cleaned_text"] or "(empty after preprocessing)", language="text")

        if not result["probabilities"].empty:
            st.subheader("Class Probabilities")
            probability_chart = result["probabilities"].set_index("Emotion")
            st.bar_chart(probability_chart["Probability"])
            st.dataframe(
                result["probabilities"].assign(
                    Probability=result["probabilities"]["Probability"].map(lambda value: f"{value:.2%}")
                ),
                use_container_width=True,
            )

        st.subheader("Influential Terms")
        terms = result["influential_terms"]
        if terms.empty:
            st.info("No influential terms were available for this input.")
        else:
            display_terms = terms.copy()
            display_terms["Estimated influence"] = display_terms["Estimated influence"].map(lambda value: f"{value:.6f}")
            st.dataframe(display_terms, use_container_width=True)


def data_explorer_page():
    render_header("Data Explorer", "Explore the new emotion dataset.")
    df = load_dataset()

    counts = df["emotion"].value_counts()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total samples", f"{len(df):,}")
    col2.metric("Emotion classes", df["emotion"].nunique())
    col3.metric("Largest class", counts.idxmax())
    col4.metric("Smallest class", counts.idxmin())

    st.subheader("Sample Data")
    st.dataframe(df[["text", "label", "emotion"]].head(25), use_container_width=True)

    st.subheader("Dataset Statistics")
    stats = pd.DataFrame(
        {
            "Metric": ["Average words", "Median words", "Average characters", "Median characters"],
            "Value": [
                round(df["word_count"].mean(), 2),
                round(df["word_count"].median(), 2),
                round(df["char_count"].mean(), 2),
                round(df["char_count"].median(), 2),
            ],
        }
    )
    st.table(stats)

    st.subheader("Data Distribution")
    chart_col, table_col = st.columns([2, 1])
    with chart_col:
        st.bar_chart(counts)
    with table_col:
        st.dataframe(counts.rename_axis("Emotion").reset_index(name="Count"), use_container_width=True)

    st.subheader("Emotion Share")
    _, pie_col, _ = st.columns([1, 1.35, 1])
    with pie_col:
        fig, ax = plt.subplots(figsize=(3.8, 3.8))
        ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 8},
        )
        ax.set_title("Emotion Distribution", fontsize=11)
        ax.axis("equal")
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)


def visualizations_page():
    render_header("Visualizations", "Dataset insights and final model performance charts.")

    visualization_assets = [
        (
            "1. Word Cloud",
            BASE_DIR / "results" / "wordcloud_by_emotion.png",
            "Frequently appearing words are shown larger, giving a quick overview of common language patterns in the emotion dataset.",
        ),
        (
            "2. Label Distribution",
            BASE_DIR / "results" / "emotion_distribution.png",
            "The dataset is imbalanced, with Joy and Sadness having the most samples while Surprise has the fewest.",
        ),
        (
            "3. Text Length Distribution",
            BASE_DIR / "results" / "text_length_distribution.png",
            "Most texts are short, which matches the social-media style input expected by the emotion detection app.",
        ),
        (
            "4. Top 20 Common Words",
            BASE_DIR / "results" / "top_20_words.png",
            "The most frequent terms reveal repeated emotional expressions and everyday language used across the dataset.",
        ),
        (
            "5. Model Comparison Chart",
            MODEL_COMPARISON_PATH,
            "DistilBERT achieved the strongest final performance, supporting its selection as the main classifier in the Streamlit app.",
        ),
        (
            "6. DistilBERT Confusion Matrix Heatmap",
            BASE_DIR / "results" / "confusion_matrix.png",
            "Strong diagonal values show accurate predictions, while off-diagonal cells highlight emotions that are sometimes confused.",
        ),
    ]

    for title, image_path, insight in visualization_assets:
        st.subheader(title)
        if image_path.exists():
            st.image(str(image_path), width=760)
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-label">Key insight</div>
                    <p class="insight-copy">{insight}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info(f"Waiting for final image: {image_path.name}")


def model_info_page():
    render_header("Model Info", "Training approach, models, and performance overview.")

    st.subheader("Emotion Classes")
    st.table(pd.DataFrame([{"Label": key, "Emotion": value} for key, value in LABEL_MAP.items()]))

    st.subheader("Training Details")
    st.markdown(
        """
        - Dataset: emotion text dataset with 416,809 labelled samples
        - Task: six-class emotion classification
        - Preprocessing: remove URLs, punctuation, digits, stop words, then apply Porter stemming
        - Emotion classifier used in the app: DistilBERT
        - Final DistilBERT performance: 0.9273 accuracy and 0.9268 weighted F1
        - Multi-language support: NLLB-200 distilled 600M translates non-English input to English before prediction
        - Supported app input languages: English, Chinese, Malay, Tamil, and Indonesian
        """
    )

    st.subheader("Traditional ML and GRU Baselines")
    st.markdown(
        """
        The final modeling notebook also includes traditional machine learning
        baselines to show the full NLP pipeline. The baseline models use
        Bag-of-Words, TF-IDF, and Word2Vec features with Naive Bayes, Logistic
        Regression, SVM, and Random Forest classifiers.

        Best traditional baseline: SVM with TF-IDF achieved 0.8961 accuracy and
        0.8956 weighted F1. The Bidirectional GRU achieved 0.9171 accuracy and
        0.9175 weighted F1. DistilBERT was selected for the Streamlit app because
        it achieved the best final performance.
        """
    )

    st.subheader("Model Performance")
    if MODEL_COMPARISON_PATH.exists():
        _, image_col, _ = st.columns([0.2, 1.6, 0.2])
        with image_col:
            st.image(str(MODEL_COMPARISON_PATH), width=760)
            st.caption("The comparison chart comes from the modeling output. The deployed app uses DistilBERT as the selected classifier.")
    else:
        st.info("Model comparison chart is not available yet.")

    st.subheader("Deployment Notes")
    st.markdown(
        """
        DistilBERT is used as the main app model because it achieved the strongest
        final evaluation performance among the tested approaches. Non-English text
        is first translated to English using Meta's pre-trained NLLB multilingual
        translation model, then the translated English text is passed into
        DistilBERT for emotion prediction.
        """
    )


PAGES = {
    "Home": home_page,
    "Text Analyzer": text_analyzer_page,
    "Data Explorer": data_explorer_page,
    "Visualizations": visualizations_page,
    "Model Info": model_info_page,
}


with st.sidebar:
    st.header("EmotionSense")
    selected_page = st.radio("Navigation", list(PAGES.keys()))
    st.caption("DistilBERT emotion model + NLLB translation")

PAGES[selected_page]()
