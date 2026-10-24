import re
import unicodedata
from pathlib import Path

from data_analyzer import (
    load_dataset,
    profile_data,
    calculate_total_revenue,
    revenue_by_product,
    revenue_by_month,
    plot_revenue_by_month,
)

from llm import generate_reply


DATASET_PATH = Path(__file__).parent / "data" / "sales.csv"


# Elenco degli strumenti che l'LLM può selezionare.
AVAILABLE_TOOLS = {
    "total_revenue",
    "product_revenue",
    "monthly_revenue",
    "monthly_chart",
    "dataset_profile",
}


def normalize_text(text: str) -> str:
    """
    Converte il testo in minuscolo e rimuove gli accenti.
    Facilita il riconoscimento delle parole italiane.
    """

    text = unicodedata.normalize("NFD", text.lower())

    return "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )


def fallback_routing(user_message: str) -> str | None:
    """
    Applica regole semplici per riconoscere le richieste
    quando l'LLM non restituisce uno strumento valido.
    """

    text = normalize_text(user_message)

    # 1. Richieste di grafici
    if any(word in text for word in (
        "grafico",
        "diagramma",
        "visualizza",
        "disegna",
    )):
        return "monthly_chart"

    # 2. Informazioni sulla struttura e sulla qualità dei dati
    if any(word in text for word in (
        "colonna",
        "colonne",
        "campo",
        "campi",
        "mancant",
        "dataset",
        "struttura",
        "tabella",
        "righe",
    )):
        return "dataset_profile"

    # 3. Analisi temporale
    if any(word in text for word in (
        "mese",
        "mesi",
        "mensil",
        "andamento",
        "andat",
        "trend",
        "nel tempo",
    )):
        return "monthly_revenue"

    # 4. Analisi e confronto dei prodotti
    if any(word in text for word in (
        "prodotto",
        "prodotti",
        "articolo",
        "articoli",
        "laptop",
        "mouse",
        "monitor",
        "keyboard",
        "redditiz",
        "porta piu soldi",
        "vende meglio",
        "classifica",
    )):
        return "product_revenue"

    # 5. Ricavi complessivi
    if any(word in text for word in (
        "totale",
        "complessiv",
        "fatturato",
        "incassat",
        "ricavo totale",
        "ricavi totali",
        "quanto abbiamo venduto",
    )):
        return "total_revenue"

    # 6. Non è stata riconosciuta una richiesta
    return None



def choose_tool(user_message: str) -> str | None:
    """
    Chiede all'LLM di selezionare uno strumento autorizzato.
    Se la risposta non è valida, prova il routing basato su regole.
    """

    routing_prompt = f"""
Classifica la richiesta dell'utente.

Devi scegliere ESATTAMENTE uno dei seguenti strumenti:

total_revenue
product_revenue
monthly_revenue
monthly_chart
dataset_profile

Significato degli strumenti:

total_revenue:
Calcola i ricavi complessivi di tutte le vendite.

product_revenue:
Calcola i ricavi per prodotto e permette di individuare
il prodotto con i ricavi più alti.

monthly_revenue:
Calcola i ricavi aggregati per mese.

monthly_chart:
Genera un grafico dei ricavi mensili.

dataset_profile:
Descrive il dataset, le colonne e i valori mancanti.

Esempi:

"Quanto sono i ricavi totali?"
Risposta: total_revenue

"Quale prodotto ha generato più ricavi?"
Risposta: product_revenue

"Mostrami i ricavi per mese."
Risposta: monthly_revenue

"Crea un grafico delle vendite."
Risposta: monthly_chart

"Ci sono valori mancanti?"
Risposta: dataset_profile

Rispondi soltanto con il nome dello strumento,
senza spiegazioni.

Richiesta dell'utente:
{user_message}
"""

    system_prompt = (
        "Sei un router di strumenti per un agente di analisi dati. "
        "Devi classificare le richieste e restituire soltanto "
        "uno dei nomi di strumenti autorizzati. "
        "Non inventare strumenti."
    )

    model_output = generate_reply(
        routing_prompt,
        system_prompt=system_prompt,
    )

    normalized_output = normalize_text(model_output)

    # Accettiamo esclusivamente strumenti presenti nella lista.
    for tool_name in sorted(AVAILABLE_TOOLS, key=len, reverse=True):
        if re.search(
            rf"\b{re.escape(tool_name)}\b",
            normalized_output,
        ):
            return tool_name

    # Se l'LLM non fornisce un nome valido, proviamo le regole.
    return fallback_routing(user_message)


def execute_tool(tool_name: str, df=None) -> str:
    """
    Esegue lo strumento selezionato usando le funzioni Python.
    """

    if df is None:
        df = load_dataset(DATASET_PATH)

    if tool_name == "total_revenue":
        total = calculate_total_revenue(df)

        return f"Ricavi totali: {total:.2f}"

    if tool_name == "product_revenue":
        summary = revenue_by_product(df)
        top_product = summary.iloc[0]

        return (
            f"Prodotto più redditizio: "
            f"{top_product['product']} "
            f"({top_product['revenue']:.2f})\n\n"
            f"Ricavi per prodotto:\n"
            f"{summary.to_string(index=False)}"
        )

    if tool_name == "monthly_revenue":
        summary = revenue_by_month(df)

        return (
            "Ricavi per mese:\n"
            f"{summary.to_string(index=False)}"
        )

    if tool_name == "monthly_chart":
        summary = revenue_by_month(df)
        chart_path = plot_revenue_by_month(summary)

        return f"Grafico creato correttamente: {chart_path}"

    if tool_name == "dataset_profile":
        profile = profile_data(df)

        missing = [
            f"- {column}: {count}"
            for column, count in profile["missing_values"].items()
            if count > 0
        ]

        missing_text = (
            "\n".join(missing)
            if missing
            else "Nessun valore mancante."
        )

        return (
            f"Righe: {profile['rows']}\n"
            f"Colonne: {profile['columns']}\n"
            f"Colonne presenti: "
            f"{', '.join(profile['column_names'])}\n\n"
            f"Valori mancanti:\n{missing_text}"
        )

    raise ValueError(f"Strumento non autorizzato: {tool_name}")


def run_agent(
    user_message: str,
    dataset_path: str | Path = DATASET_PATH,
) -> str:
    """
    Coordina la scelta dello strumento e la sua esecuzione
    sul dataset specificato.
    """

    if not isinstance(user_message, str) or not user_message.strip():
        return "Inserisci una domanda."

    tool_name = choose_tool(user_message)

    if tool_name is None:
        return (
            "Non sono riuscito a identificare lo strumento adatto. "
            "Prova a specificare se vuoi conoscere i ricavi totali, "
            "analizzare i prodotti, vedere i dati mensili "
            "oppure generare un grafico."
        )

    df = load_dataset(dataset_path)

    result = execute_tool(tool_name, df=df)

    return (
        f"Strumento selezionato: {tool_name}\n\n"
        f"{result}"
    )


if __name__ == "__main__":
    print("AI Data Analyst Agent")
    print("Scrivi 'esci' per terminare.")

    while True:
        question = input("\nTu: ").strip()

        if question.lower() in {"esci", "exit", "quit"}:
            print("Agente terminato.")
            break

        try:
            answer = run_agent(question)
            print(f"\nAgente:\n{answer}")

        except (ValueError, FileNotFoundError) as exc:
            print(f"\nErrore: {exc}")