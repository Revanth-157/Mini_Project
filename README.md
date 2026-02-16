# Mini_Project
We use HuBERT to extract acoustic features from speech, capturing tone and prosody for emotion classification. For text, RoBERTa generates contextual embeddings to detect emotional cues. Both outputs are classified into emotions like happy, sad, or angry for multimodal emotion detection.
Here’s a **brief and clear description** you can use for your emotion detection project:

---

### 1️⃣ Using **HuBERT** for Emotion Detection from Speech

HuBERT (Hidden-Unit BERT) is a **self-supervised speech representation model** developed by **Meta AI**. It learns powerful audio features by predicting masked portions of speech signals.

In our project:

* We use **HuBERT as a feature extractor** for raw audio.
* The input speech waveform is passed into HuBERT.
* HuBERT generates **contextual speech embeddings** that capture tone, pitch, rhythm, and acoustic patterns.
* These embeddings are fed into a **classification layer (e.g., fully connected + softmax)** to predict emotions like *happy, sad, angry, neutral*, etc.

**Why HuBERT?**

* Captures prosody (intonation, stress, pitch).
* Works well even with limited labeled emotion data.
* Pretrained on large-scale speech datasets, so it generalizes well.

---

### 2️⃣ Using **RoBERTa** for Emotion Detection from Text

RoBERTa (Robustly Optimized BERT Approach) is an improved version of BERT developed by **Facebook AI**. It is trained on large-scale text corpora and generates deep contextual embeddings.

In our project:

* Text input (transcripts or chat messages) is tokenized.
* The tokens are passed into RoBERTa.
* RoBERTa generates contextual word representations.
* The `[CLS]` token embedding (or pooled output) is passed to a **classification head**.
* The model predicts emotions such as *joy, sadness, fear, anger*, etc.

**Why RoBERTa?**

* Strong contextual understanding.
* Better performance than standard BERT.
* Handles subtle emotional cues in language.

---

### 🔹 Combined System (Multimodal Emotion Detection)

* HuBERT → extracts **acoustic emotional cues**
* RoBERTa → extracts **semantic emotional cues**
* Outputs can be:

  * Used independently, or
  * Combined (concatenated embeddings) for improved multimodal emotion detection accuracy.

---

If you want, I can also give:

* A 5-mark exam-ready answer
* A technical explanation for report
* Or architecture diagram explanation 🎯
