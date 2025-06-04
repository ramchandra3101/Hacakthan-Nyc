# Hacakthan-Nyc

This repository contains the project we built during the **Microsoft Hack Night** in New York City.

## Project Overview

This AI-powered music recommendation app interprets natural language mood prompts like:

- “I’m sad but want something uplifting”
- “Play something calm while I study”

…and recommends Spotify tracks accordingly.


### The project includes the following components:

- **frontend.py**: A Python script that handles the frontend interface or user interactions built with streamlit.

- **main.py**: The main execution script that orchestrates the overall workflow of the application.


## Tech Stack

- **Python**
- **FriendliAI** – LLM inference
- **Weaviate** – Semantic vector search
- **Spotify API** – Track sourcing
- **Comet** – Observability and experiment tracking


## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/ramchandra3101/Hacakthan-Nyc.git
cd Hacakthan-Nyc
```


### 2. Create a virtual environment (optional)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

> Note: If requirements.txt is missing, install packages manually based on the code.

```bash
pip install -r requirements.txt
```


### 4. Run the app

```bash
python main.py
```


### Highlights

**🏆 Winner – Best Use of FriendliAI**

> We used semantic search and mood-aware LLMs to deliver truly personalized music discovery. Vector search via Weaviate enabled matching based on meaning instead of keywords.


### License

This project is licensed under the MIT License.

## Contributors

- Ramachandra Yerramsetti
- Sai Kiran Belana
- Hiral Choksi

