import json
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
    "sales_overview",
}

def parse_tool_call(model_output: str) -> dict | None:
    """
    Interpreta e valida il JSON generato dal modello.

    Accetta strumenti autorizzati e, per product_revenue,
    permette un argomento opzionale chiamato 'product'.
    """

    try:
        cleaned_output = model_output.strip()

        # Rimuove gli eventuali delimitatori Markdown.
        if cleaned_output.startswith("```"):
            lines = cleaned_output.splitlines()

            # Rimuove la prima riga, ad esempio ```json.
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]

            # Rimuove la chiusura ```.
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            cleaned_output = "\n".join(lines).strip()

        payload = json.loads(cleaned_output)

    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(payload, dict):
        return None

    if set(payload.keys()) != {"tool", "arguments"}:
        return None

    tool_name = payload["tool"]
    arguments = payload["arguments"]

    if not isinstance(tool_name, str):
        return None

    if tool_name not in AVAILABLE_TOOLS:
        return None

    if not isinstance(arguments, dict):
        return None

    # Tutti gli strumenti possono essere chiamati senza argomenti.
    if arguments == {}:
        return {
            "tool": tool_name,
            "arguments": {},
        }

    # Soltanto product_revenue accetta l'argomento product.
    if tool_name == "product_revenue":
        if set(arguments.keys()) != {"product"}:
            return None

        product = arguments["product"]

        if not isinstance(product, str) or not product.strip():
            return None

        return {
            "tool": tool_name,
            "arguments": {
                "product": product.strip(),
            },
        }

    # Gli altri strumenti non accettano argomenti in questa versione.
    return None


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

    # Richieste che combinano analisi per prodotto e per mese.
    product_terms = (
        "prodotto",
        "prodotti",
        "articolo",
        "articoli",
        "laptop",
        "mouse",
        "monitor",
        "keyboard",
    )

    month_terms = (
        "mese",
        "mesi",
        "mensil",
        "andamento",
        "trend",
        "nel tempo",
    )

    has_product = any(term in text for term in product_terms)
    has_month = any(term in text for term in month_terms)

    if has_product and has_month:
        return "sales_overview"

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



def choose_tool(user_message: str) -> dict | None:
    """
    Chiede all'LLM di scegliere uno strumento e i suoi argomenti.
    Valida il JSON e utilizza il fallback se la risposta non è valida.
    """

    routing_prompt = f"""
Classifica la richiesta dell'utente.

Gli strumenti disponibili sono:

sales_overview:
Calcola i ricavi per prodotto e i ricavi per mese,
restituendo entrambi i riepiloghi.

total_revenue:
Calcola i ricavi complessivi.

product_revenue:
Calcola i ricavi per prodotto.
Può ricevere l'argomento opzionale "product".
Se l'utente chiede i ricavi di un prodotto specifico,
inserisci il suo nome nell'argomento.
Se chiede una classifica o il prodotto più redditizio,
usa arguments vuoto.

monthly_revenue:
Calcola i ricavi per mese.

monthly_chart:
Genera un grafico dei ricavi mensili.

dataset_profile:
Descrive il dataset e i valori mancanti.

Restituisci esclusivamente un JSON valido,
senza spiegazioni e senza Markdown.

Esempio per una richiesta generica:
{{"tool": "product_revenue", "arguments": {{}}}}

Esempio per una richiesta relativa a un prodotto specifico:
{{"tool": "product_revenue", "arguments": {{"product": "Laptop"}}}}

Altri esempi:

"Analizza i ricavi per prodotto e per mese."
{{"tool": "sales_overview", "arguments": {{}}}}

"Quanto sono i ricavi totali?"
{{"tool": "total_revenue", "arguments": {{}}}}

"Mostrami i ricavi per mese."
{{"tool": "monthly_revenue", "arguments": {{}}}}

"Crea un grafico delle vendite."
{{"tool": "monthly_chart", "arguments": {{}}}}

"Ci sono valori mancanti?"
{{"tool": "dataset_profile", "arguments": {{}}}}


Richiesta dell'utente:
{user_message}
"""

    system_prompt = (
        "Sei un router per un agente di analisi dati. "
        "Restituisci esclusivamente JSON valido nel formato richiesto. "
        "Utilizza soltanto gli strumenti autorizzati. "
        "Non inventare argomenti."
    )

    model_output = generate_reply(
        routing_prompt,
        system_prompt=system_prompt,
    )

    tool_call = parse_tool_call(model_output)

    # Non accettiamo un prodotto specifico se l'utente
    # non lo ha menzionato esplicitamente nella domanda.
    if tool_call is not None and tool_call["tool"] == "product_revenue":
        product = tool_call["arguments"].get("product")

        if product:
            normalized_question = normalize_text(user_message)
            normalized_product = normalize_text(product)

            product_was_mentioned = re.search(
                rf"\b{re.escape(normalized_product)}\b",
                normalized_question,
            )

            if not product_was_mentioned:
                tool_call["arguments"] = {}


    if tool_call is not None:
        return tool_call

    # Il fallback mantiene il comportamento precedente.
    fallback_tool = fallback_routing(user_message)

    if fallback_tool is None:
        return None

    return {
        "tool": fallback_tool,
        "arguments": {},
    }


def execute_tool(
    tool_name: str,
    df=None,
    arguments: dict | None = None,
) -> str:
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

        product_name = (arguments or {}).get("product")

        # Se è stato indicato un prodotto specifico,
        # restituiamo soltanto i suoi ricavi.
        if product_name:
            matches = summary[
                summary["product"].astype(str).str.casefold()
                == product_name.casefold()
            ]

            if matches.empty:
                return f"Prodotto non trovato: {product_name}"

            row = matches.iloc[0]

            return (
                f"Ricavi per {row['product']}: "
                f"{row['revenue']:.2f}"
            )

        # Altrimenti manteniamo il comportamento precedente.
        top_product = summary.iloc[0]

        return (
            f"Prodotto più redditizio: "
            f"{top_product['product']} "
            f"({top_product['revenue']:.2f})\n\n"
            f"Ricavi per prodotto:\n"
            f"{summary.to_string(index=False)}"
        )

        return (
            f"Prodotto più redditizio: "
            f"{top_product['product']} "
            f"({top_product['revenue']:.2f})\n\n"
            f"Ricavi per prodotto:\n"
            f"{summary.to_string(index=False)}"
        )

    if tool_name == "sales_overview":
        # Primo passaggio: calcolo dei ricavi per prodotto.
        product_summary = revenue_by_product(df)

        # Secondo passaggio: calcolo dei ricavi per mese.
        monthly_summary = revenue_by_month(df)

        top_product = product_summary.iloc[0]

        return (
            "=== RICAVI PER PRODOTTO ===\n"
            f"{product_summary.to_string(index=False)}\n\n"
            f"Prodotto con i ricavi più alti: "
            f"{top_product['product']} "
            f"({top_product['revenue']:.2f})\n\n"
            "=== RICAVI PER MESE ===\n"
            f"{monthly_summary.to_string(index=False)}"
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
    Coordina la selezione dello strumento e la sua esecuzione.
    """

    if not isinstance(user_message, str) or not user_message.strip():
        return "Inserisci una domanda."

    tool_call = choose_tool(user_message)

    if tool_call is None:
        return (
            "Non sono riuscito a identificare lo strumento adatto. "
            "Prova a specificare quale analisi vuoi eseguire."
        )

    tool_name = tool_call["tool"]
    arguments = tool_call["arguments"]

    df = load_dataset(dataset_path)

    result = execute_tool(
        tool_name,
        df=df,
        arguments=arguments,
    )

    return (
        f"Strumento selezionato: {tool_name}\n"
        f"Argomenti: {arguments}\n\n"
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