# Capstone 14 — Placement Readiness Bot

Ek hi conversational agent, do tools ke saath:
1. **predict_placement** → trained `DecisionTreeClassifier` model se likelihood
2. **search_policy** → real placement-policy document se grounded (RAG) answers

Poora project already bana hua hai is folder mein. Neeche step-by-step chalne ka tareeka hai.

---

## 0. Folder structure

```
placement_readiness_bot/
├── requirements.txt
├── .env.example
├── data/
│   └── placement_data.csv          (generate karoge - step 2)
├── policy_docs/
│   └── placement_policy.txt        (sample policy - apna real doc yahan daal sakte ho)
├── model/
│   ├── placement_model.pkl         (train karoge - step 3)
│   └── policy_faiss_index/         (auto-generate hoga pehli baar app chalane par)
├── charts/
│   ├── feature_importance.png
│   └── dataset_overview.png
└── src/
    ├── generate_data.py
    ├── train_model.py
    ├── rag_utils.py
    ├── tools.py
    ├── agent.py
    └── app.py
```

---

## 1. Python venv setup

```bash
cd placement_readiness_bot
python3 -m venv venv

# Activate:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

> Pehli baar `sentence-transformers` install hone mein thoda time lagega
> (embedding model download hota hai) — internet chahiye.

---

## 2. API key set karo (FREE option — Groq)

`.env.example` ko copy karke `.env` banao:

```bash
cp .env.example .env
```

Fir `.env` file open karke:

1. `LLM_PROVIDER=groq` (already default hai, ye rehne do)
2. Groq ki **free** API key lo: https://console.groq.com/keys
   (Google/GitHub se login karo, "Create API Key" pe click karo — koi card nahi maangega)
3. `.env` mein `GROQ_API_KEY=` ke aage apni key paste karo

```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

> **OpenAI use karna ho** (agar billing already set up hai): `.env` mein
> `LLM_PROVIDER=openai` kar do aur `OPENAI_API_KEY=` mein apni key daalo.
> Baaki code automatically switch ho jayega — kuch aur change nahi karna.

---

## 3. Synthetic dataset banao

```bash
python src/generate_data.py
```

Ye `data/placement_data.csv` banayega (1200 students, features: cgpa,
backlogs, internships, projects, coding_score, communication_score,
attendance_percent, placed).

**Real project ke liye:** apne college/seniors ka real anonymized placement
data isi column format mein `data/placement_data.csv` mein daal do — bas
column names match hone chahiye.

---

## 4. Model train karo

```bash
python src/train_model.py
```

Ye:
- `model/placement_model.pkl` save karega (DecisionTreeClassifier)
- `charts/feature_importance.png` aur `charts/dataset_overview.png` banayega
- Terminal mein accuracy/classification report dikhayega (~80% accuracy expected)

---

## 5. Policy document check karo

`policy_docs/placement_policy.txt` mein ek sample eligibility policy already
likhi hui hai (CGPA cutoff, backlog rules, attendance, debarment, dress code,
documentation, etc.).

**Real project ke liye:** apne college ka actual placement policy PDF/doc
lekar uska text isi file mein replace kar do (ya `rag_utils.py` mein PDF
loader use kar lo — `PyPDFLoader` import karke).

---

## 6. App chalao

```bash
streamlit run src/app.py
```

Browser mein khulega. Left sidebar mein charts dikhenge, chat box mein
questions puch sakte ho.

---

## 7. Test cases (PDF ke test cases ke hisaab se try karo)

| # | Try this in chat | Expect |
|---|---|---|
| TC1 | "What's my placement probability? CGPA 8.2, 0 backlogs, 2 internships, 3 projects, coding 80, communication 75, attendance 90%" | Sirf `predict_placement` call hoga |
| TC2 | "What is the minimum CGPA required to be eligible for placements?" | Sirf `search_policy` call hoga |
| TC3 | "Am I eligible for placements AND likely to get placed? CGPA 6.5, 1 backlog, 1 internship, 2 projects, coding 60, communication 65, attendance 80%" | Dono tools call honge, ek combined answer |
| TC4 | "What stipend does Google pay for internships?" | Honest fallback ("document doesn't cover this") |
| TC5 | Same input do baar, agent ka answer aur seedha `python -c` se model chalake compare karo | Values match |
| TC6 | "Hi, how are you?" | Koi tool call nahi, seedha reply |
| TC7 | Pehle ek prediction poochho, phir "what did you just tell me my chances were?" | Purana context yaad rakh ke sahi answer de |

---

## 8. Streamlit Community Cloud pe deploy (optional)

1. Is folder ko GitHub repo mein push karo (`git init`, `git add .`,
   `git commit -m "capstone 14"`, `git push`)
2. [share.streamlit.io](https://share.streamlit.io) pe jaake repo connect karo
3. Main file path: `src/app.py`
4. "Secrets" mein `OPENAI_API_KEY` add karo

---

## 9. Stretch features (agar time bache)

- **Combined synthesis:** `agent.py` ka system prompt already dono tools ke
  results ko ek jawab mein combine karne ko bolta hai — check karo TC3 mein
  ye ho raha hai ya nahi.
- **Logging:** `tools.py` ke andar har tool call ke end mein ek SQLite table
  (`sqlite3` module, built-in) mein query + result + timestamp insert kar do.

---

## Common issues

- **"No OPENAI_API_KEY found"** → `.env` file check karo, `python-dotenv`
  install hai ya nahi.
- **FAISS/sentence-transformers install slow/fail** → stable internet chahiye
  pehli baar; ya `faiss-cpu` version apne OS ke hisaab se adjust karo.
- **Model file not found** → pehle `train_model.py` run karna zaroori hai
  `app.py` chalane se pehle.
