# ♻️ WasteSort AI

A web app that uses AI to classify waste into 6 categories and tells you how to dispose of it correctly. Just upload a photo and it does the rest.

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange) ![Accuracy](https://img.shields.io/badge/Val%20Accuracy-96.6%25-brightgreen)

---

## What it does

Upload a photo of any waste item and the app will:
- Classify it into one of 6 categories: **Cardboard, Glass, Metal, Paper, Plastic, Trash**
- Tell you which bin to use
- Give you a disposal tip
- Show a confidence breakdown across all categories

---

## Setup Instructions (Windows)

Follow every step in order. Don't skip anything.

---

### Step 1 — Install Python

1. Go to https://www.python.org/downloads/
2. Click **Download Python 3.x.x** (the big yellow button)
3. Run the installer
4. ⚠️ On the first screen, tick **"Add Python to PATH"** before clicking Install
5. Click **Install Now**

To verify it worked, open Command Prompt and type:
```
python --version
```
You should see something like `Python 3.11.x`

---

### Step 2 — Install Git

1. Go to https://git-scm.com/download/win
2. Download and run the installer
3. Click Next on every screen — the defaults are fine

To verify it worked, open Command Prompt and type:
```
git --version
```
You should see something like `git version 2.x.x`

---

### Step 3 — Clone the repository

This downloads all the project files to your computer.

Open **Command Prompt** (search for `cmd` in the Start menu) and run:

```
git clone https://github.com/tejaswibteggihalli/AI_EL.git
```

Then move into the project folder:

```
cd AI_EL
```

---

### Step 4 — Create a virtual environment

A virtual environment keeps this project's dependencies separate from the rest of your computer.

```
python -m venv .venv
```

---

### Step 5 — Activate the virtual environment

```
.venv\Scripts\activate
```

You'll know it worked when you see `(.venv)` appear at the start of the line in your terminal, like this:

```
(.venv) C:\Users\YourName\AI_EL>
```

⚠️ You need to do this step every time you open a new terminal to run the app.

---

### Step 6 — Install the required libraries

```
pip install streamlit torch torchvision Pillow numpy
```

This will take a few minutes. Wait for it to finish completely.

---

### Step 7 — Run the app

```
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`

If it doesn't open, copy that link and paste it into your browser manually.

---

### Step 8 — Use the app

1. Click **Browse files** or drag and drop a photo of waste
2. The app will show you the category, confidence, disposal tip, and a bar chart
3. To stop the app, go back to the terminal and press `Ctrl + C`

---

## Running the app again later

Every time you want to use the app after the first setup, just do Steps 5 and 7:

```
cd AI_EL
.venv\Scripts\activate
streamlit run app.py
```

---

## Project Structure

```
AI_EL/
├── app.py              ← the web app
├── best_model.pth      ← trained AI model weights
├── requirements.txt    ← list of dependencies
└── README.md           ← this file
```

---

## Troubleshooting

**`python` is not recognized**
→ Python is not installed or not added to PATH. Redo Step 1 and make sure you tick "Add Python to PATH".

**`git` is not recognized**
→ Git is not installed. Redo Step 2.

**`streamlit` is not recognized**
→ Your virtual environment is not activated. Run `.venv\Scripts\activate` first.

**The browser doesn't open**
→ Manually go to `http://localhost:8501` in your browser.

**Model file not found error**
→ Make sure `best_model.pth` is in the same folder as `app.py`. It should already be there after cloning.

---

## Model Details

- **Architecture:** EfficientNet-B0 (pretrained on ImageNet, fine-tuned)
- **Dataset:** 13,901 images across 6 waste categories
- **Training:** 2-phase training — head only for 5 epochs, full fine-tune for 15 epochs
- **Validation Accuracy:** 96.6%

---

## Built With

- [PyTorch](https://pytorch.org/) — model training and inference
- [Streamlit](https://streamlit.io/) — web app framework
- [EfficientNet-B0](https://arxiv.org/abs/1905.11946) — CNN architecture
