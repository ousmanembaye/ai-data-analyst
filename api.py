import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import run_agent


app = FastAPI(
    title="AI Data Analyst API",
    description="API for AI-powered data analysis",
    version="1.0.0",
)


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