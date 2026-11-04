import logging

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from agent import run_agent

import os
import tempfile
from pathlib import Path


app = FastAPI(
    title="AI Data Analyst API",
    description="API for AI-powered data analysis",
    version="1.0.0",
)

ALLOWED_EXTENSIONS = {".csv", ".xlsx"}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# Modello che definisce la struttura della richiesta.
class QuestionRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=500,
        description="Domanda da inviare all'agente AI",
    )


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "AI Data Analyst API is running",
    }


@app.post("/analyze")
def analyze_question(request: QuestionRequest):
    """
    Riceve una domanda HTTP e la inoltra all'AI Agent.
    """

    try:
        answer = run_agent(request.question)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        logging.exception("Dataset predefinito non trovato")

        raise HTTPException(
            status_code=500,
            detail="Dataset predefinito non disponibile.",
        ) from exc

    except Exception as exc:
        logging.exception("Errore durante l'esecuzione dell'agente")

        raise HTTPException(
            status_code=500,
            detail="Errore interno durante l'analisi.",
        ) from exc

    return {
        "status": "success",
        "question": request.question,
        "answer": answer,
    }

@app.post("/analyze-upload")
def analyze_uploaded_file(
    question: str = Form(..., min_length=1, max_length=500),
    file: UploadFile = File(...)
):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Formato non supportato. Carica un file CSV o XLSX."
        )

    temp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_file_path = temp_file.name
            total_size = 0

            while True:
                chunk = file.file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File troppo grande. Dimensione massima consentita: 10 MB."
                    )

                temp_file.write(chunk)

        answer = run_agent(
            question,
            dataset_path=temp_file_path
        )

        return {
            "status": "success",
            "filename": file.filename,
            "question": question,
            "answer": answer
        }

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)