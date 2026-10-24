# AI Data Analyst Agent

Agente AI che permette di analizzare dataset CSV ed Excel attraverso domande in linguaggio naturale.

## Funzionalità

* Caricamento di file CSV ed Excel (`.xlsx`).
* Calcolo dei ricavi totali.
* Analisi e confronto dei ricavi per prodotto.
* Analisi dei ricavi mensili.
* Individuazione dei valori mancanti.
* Generazione di grafici.
* Interfaccia web con Gradio.
* Test automatici con `unittest`.

## Tecnologie

* Python
* Pandas
* Matplotlib
* PyTorch
* Hugging Face Transformers
* Qwen2.5-0.5B-Instruct
* Gradio

## Architettura

Il progetto combina un modello linguistico locale con strumenti Python per analizzare i dati.

**Domanda dell'utente → LLM → Selezione dello strumento → Analisi Python → Risultato**

I calcoli vengono eseguiti da Python, anziché essere affidati direttamente al modello linguistico.

## Installazione

Crea e attiva un ambiente virtuale, quindi installa le dipendenze:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Avvio

Esegui:

```powershell
python app.py
```

Apri nel browser l'indirizzo locale mostrato nel terminale, normalmente `http://127.0.0.1:7860`.

Al primo avvio potrebbe essere necessario scaricare il modello linguistico da Hugging Face.

## Test

Per eseguire tutti i test:

```powershell
python -m unittest discover -s tests -v
```

## Dataset di esempio

Il progetto include `data/sales.csv`, un piccolo dataset di vendite utilizzato per verificare le funzionalità dell'agente.

## Obiettivo

Realizzare un agente AI che combini modelli linguistici, strumenti Python e analisi deterministica dei dati in un'applicazione interattiva.
