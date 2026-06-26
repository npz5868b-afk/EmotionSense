# EmotionSense

EmotionSense is a Streamlit web application for multi-class emotion detection in text. It classifies short text into six emotion classes using the team's selected DistilBERT model. For non-English input, the app uses a pre-trained OPUS-MT multilingual-to-English translation model through Hugging Face Inference API to translate the text to English before emotion prediction.

The app follows the final modeling notebook preprocessing flow: remove URLs, punctuation, digits, and stop words, then apply Porter stemming before passing text into the emotion classifier.

## Emotion Classes

```python
0 = Sadness
1 = Joy
2 = Love
3 = Anger
4 = Fear
5 = Surprise
```

## Project Structure

```text
EmotionSense/
|-- app.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- text.csv
|-- models/
|   |-- distilbert_emotion/
|-- results/
|   |-- model_comparison.png
|   |-- wordcloud_by_emotion.png
|   |-- emotion_distribution.png
|   |-- text_length_distribution.png
|   |-- top_20_words.png
|   |-- confusion_matrix.png
|-- notebooks/
    |-- final project notebooks
```

## How To Run

Use Python 3.11 for TensorFlow compatibility.

```powershell
cd "C:\Users\npz58\OneDrive\Desktop\NLP Project\App"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If dependencies are not installed yet:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Pages

- Home/About: project overview, problem statement, usage guide, and team roles
- Text Analyzer: text input, optional translation, prediction, confidence score, class probabilities, and influential terms
- Data Explorer: sample data, dataset statistics, class counts, bar chart, and pie chart
- Visualizations: final dataset insights and model performance charts with short explanations
- Model Info: model explanation, training details, performance chart, and deployment notes

## Visualization Files

Place the final visualization images in the `results/` folder using these names:

```text
wordcloud_by_emotion.png
emotion_distribution.png
text_length_distribution.png
top_20_words.png
confusion_matrix.png
model_comparison.png
```

These files are displayed on the Visualizations page. `model_comparison.png` and
`confusion_matrix.png` should match the final modeling results.

## Notes

- Main model: DistilBERT
- Final DistilBERT performance: 0.9273 accuracy and 0.9268 weighted F1
- Traditional ML baselines are included in the modeling notebook for comparison: Naive Bayes, Logistic Regression, SVM, and Random Forest with BoW, TF-IDF, and Word2Vec features
- Best traditional baseline: SVM with TF-IDF achieved 0.8961 accuracy and 0.8956 weighted F1
- Bidirectional GRU achieved 0.9171 accuracy and 0.9175 weighted F1
- Translation model: Helsinki-NLP/opus-mt-mul-en through Hugging Face Inference API
- Multi-language support: Chinese, Malay, Tamil, and Indonesian input can be translated to English before prediction
- Deployment note: Hugging Face API runs the translation model externally so Streamlit Cloud does not need to load a translation model into memory
