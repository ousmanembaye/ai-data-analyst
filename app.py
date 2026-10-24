from pathlib import Path

import gradio as gr

from agent import run_agent


PROJECT_DIR = Path(__file__).parent
CHART_PATH = PROJECT_DIR / "output" / "revenue_by_month.png"


def answer_question(
    uploaded_file: str | None,
    question: str,
) -> tuple[str, str | None]:
    """
    Riceve un file caricato e una domanda.
    Restituisce la risposta dell'agente e, se generato,
    il grafico mensile.
    """

    if not uploaded_file:
        return "Carica prima un file CSV o Excel.", None

    if not isinstance(question, str) or not question.strip():
        return "Inserisci una domanda.", None

    try:
        answer = run_agent(
            question,
            dataset_path=uploaded_file,
        )

        chart = None

        if "Strumento selezionato: monthly_chart" in answer:
            if CHART_PATH.is_file():
                chart = str(CHART_PATH)

        return answer, chart

    except Exception as exc:
        return (
            f"Errore durante l'analisi: {exc}",
            None,
        )


with gr.Blocks(title="AI Data Analyst") as demo:

    gr.Markdown(
        """
        # AI Data Analyst

        Carica un dataset e fai domande in linguaggio naturale.

        Puoi usare file CSV e file Excel `.xlsx`.
        """
    )

    uploaded_file = gr.File(
        label="Carica il dataset",
        file_types=[".csv", ".xlsx"],
        type="filepath",
    )

    question = gr.Textbox(
        label="La tua domanda",
        placeholder=(
            "Esempio: Quale prodotto ha generato più ricavi?"
        ),
        lines=2,
    )

    analyze_button = gr.Button(
        "Analizza",
        variant="primary",
    )

    answer = gr.Textbox(
        label="Risultato",
        lines=12,
        interactive=False,
    )

    chart = gr.Image(
        label="Grafico generato",
        interactive=False,
    )

    analyze_button.click(
        fn=answer_question,
        inputs=[uploaded_file, question],
        outputs=[answer, chart],
    )

    question.submit(
        fn=answer_question,
        inputs=[uploaded_file, question],
        outputs=[answer, chart],
    )


if __name__ == "__main__":
    demo.launch()