# 📝 Text Summarizer

A Streamlit app that summarizes pasted text, `.txt` files, or PDFs using BART models. Long documents are split into chunks, summarized, and merged into one concise summary. It runs locally and needs no API key.

## Features

- Paste text or upload a `.txt` / `.pdf` file
- Two models: DistilBART (faster, lighter) and BART Large (better quality)
- Three summary lengths: Short, Medium, Long
- Handles long documents by chunking, then re-summarizing the merged result
- Side-by-side view of the original and the summary, with word counts and percent shorter
- Download the summary as a `.txt` file
- Custom CSS styling

## Requirements

- Python 3.9 or newer
- `streamlit`
- `transformers`
- `torch`
- `pypdf`

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install streamlit transformers torch pypdf
```

## Run

```bash
streamlit run summarizer.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Usage

1. Choose a model and summary length in the sidebar.
2. Select **Paste text** or **Upload file**.
3. Add at least 50 words of text.
4. Click **⚡ Summarize**.
5. Read the summary next to the original, or click **⬇️ Download summary**.

## How it works

1. The text is split into chunks of about 600 words, which stays under BART's 1024-token input limit.
2. Each chunk is summarized with beam search (`num_beams=4`).
3. The chunk summaries are joined together.
4. If the joined summary is still longer than one chunk, it is summarized once more.

## Notes

- **First run:** the selected model is downloaded from Hugging Face and cached, so it needs an internet connection and some disk space. Later runs load from the cache.
- **Speed:** DistilBART is noticeably faster on CPU. BART Large gives better quality but is slower.
- **PDFs:** text is extracted with `pypdf`. Scanned PDFs without a text layer will not work, since there is no OCR.
- **Language:** the models are trained on English news articles, so English text works best.
- **Theme:** the CSS is designed for a light theme. To force it, create `.streamlit/config.toml`:

```toml
[theme]
base = "light"
```

## Project structure

```
.
├── summarizer.py    # The Streamlit app
└── README.md
```

## Tech stack

- [Streamlit](https://streamlit.io/) for the UI
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) with `facebook/bart-large-cnn` and `sshleifer/distilbart-cnn-12-6`
- [PyTorch](https://pytorch.org/) as the model backend
- [pypdf](https://pypdf.readthedocs.io/) for PDF text extraction
