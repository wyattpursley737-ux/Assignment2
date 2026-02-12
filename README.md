[README.txt](https://github.com/user-attachments/files/25248099/README.txt)
FAQ Chatbot Project (Fixed)

Quick start (Windows / macOS / Linux):

1) Create and activate a virtual environment (optional but recommended)

2) Install dependencies:
   pip install -r requirements.txt

3) Set your API key:
   - Windows (PowerShell):  setx OPENAI_API_KEY "YOUR_KEY"
   - macOS/Linux (bash):    export OPENAI_API_KEY="YOUR_KEY"

4) Build the vector store (run from project root):
   python -m src.ingest

5) Run the chatbot:
   python -m src.chatbot

6) Generate evaluation run logs (optional):
   python -m src.eval_runlogs

Notes:
- The dataset is in: data/instant_pot_duo_faq.csv
- The FAISS index will be written to: vectorstore_faiss/
