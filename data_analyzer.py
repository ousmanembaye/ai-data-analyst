from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

def load_csv(file_path: str | Path) -> pd.DataFrame:
    """
    Carica un file CSV e restituisce i dati sotto forma di DataFrame.

    Solleva eccezioni comprensibili se il file non esiste,
    è vuoto oppure presenta errori di lettura.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File CSV non trovato: {path}")

    try:
        df = pd.read_csv(path)

    except pd.errors.EmptyDataError as exc:
        raise ValueError("Il file CSV è vuoto.") from exc

    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Impossibile interpretare il file CSV: {path}"
        ) from exc

    if df.empty:
        raise ValueError("Il CSV non contiene righe di dati.")

    return df

def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """
    Carica un dataset CSV o Excel (.xlsx).
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"File non trovato: {path}"
        )

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return load_csv(path)

    if suffix == ".xlsx":
        try:
            df = pd.read_excel(path, engine="openpyxl")

        except Exception as exc:
            raise ValueError(
                f"Impossibile leggere il file Excel: {path}"
            ) from exc

        if df.empty:
            raise ValueError(
                "Il file Excel non contiene righe di dati."
            )

        return df

    raise ValueError(
        "Formato non supportato. "
        "Carica un file CSV (.csv) o Excel (.xlsx)."
    )


def profile_data(df: pd.DataFrame) -> dict:
    """
    Analizza la struttura del DataFrame e restituisce
    un profilo generale del dataset.
    """
    missing_values = df.isna().sum()

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    other_columns = df.select_dtypes(
        exclude="number"
    ).columns.tolist()

    profile = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "missing_values": {
            str(column): int(count)
            for column, count in missing_values.items()
        },
        "numeric_columns": numeric_columns,
        "other_columns": other_columns,
    }

    return profile

def add_revenue_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola i ricavi di ogni vendita e aggiunge
    una colonna chiamata 'revenue'.
    """
    required_columns = {"product", "quantity", "unit_price"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Colonne necessarie mancanti: {sorted(missing_columns)}"
        )

    result = df.copy()

    for column in ("quantity", "unit_price"):
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

        if result[column].isna().any():
            raise ValueError(
                f"La colonna '{column}' contiene "
                "valori mancanti o non numerici."
            )

    if result["product"].isna().any():
        raise ValueError("La colonna 'product' contiene valori mancanti.")

    result["revenue"] = result["quantity"] * result["unit_price"]

    return result


def calculate_total_revenue(df: pd.DataFrame) -> float:
    """
    Calcola il totale dei ricavi di tutte le vendite.
    """
    sales = add_revenue_column(df)

    return float(sales["revenue"].sum())


def revenue_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola i ricavi complessivi per ogni prodotto,
    ordinandoli dal più redditizio al meno redditizio.
    """
    sales = add_revenue_column(df)

    summary = (
        sales.groupby("product", as_index=False)["revenue"]
        .sum()
        .sort_values(by="revenue", ascending=False)
        .reset_index(drop=True)
    )

    return summary

def revenue_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola i ricavi complessivi per ogni mese,
    ordinando i risultati cronologicamente.
    """
    if "date" not in df.columns:
        raise ValueError("La colonna 'date' non esiste.")

    sales = add_revenue_column(df)

    dates = pd.to_datetime(
        sales["date"],
        format="mixed",
        errors="coerce",
    )
    
    if dates.isna().any():
        raise ValueError(
            "La colonna 'date' contiene date mancanti o non valide."
        )

    sales["month"] = dates.dt.to_period("M").astype(str)

    summary = (
        sales.groupby("month", as_index=False)["revenue"]
        .sum()
        .sort_values(by="month")
        .reset_index(drop=True)
    )

    return summary

def plot_revenue_by_month(
    monthly_summary: pd.DataFrame,
    output_dir: str | Path | None = None
) -> Path:
    """
    Genera un grafico dei ricavi mensili e lo salva come PNG.
    """

    required_columns = {"month", "revenue"}
    missing_columns = required_columns - set(monthly_summary.columns)

    if missing_columns:
        raise ValueError(
            f"Colonne necessarie mancanti: {sorted(missing_columns)}"
        )

    if monthly_summary.empty:
        raise ValueError("Non ci sono dati da visualizzare.")

    # Creiamo la cartella in cui salvare il grafico.
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Creiamo il grafico.
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        monthly_summary["month"],
        monthly_summary["revenue"],
        marker="o"
    )

    ax.set_title("Andamento dei ricavi mensili")
    ax.set_xlabel("Mese")
    ax.set_ylabel("Ricavi")

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Evitiamo che gli elementi del grafico si sovrappongano.
    fig.tight_layout()

    # Salviamo il grafico come immagine.
    chart_path = output_dir / "revenue_by_month.png"
    fig.savefig(chart_path, dpi=150)

    # Liberiamo la memoria occupata dalla figura.
    plt.close(fig)

    return chart_path


if __name__ == "__main__":
    dataset_path = Path(__file__).parent / "data" / "sales.csv"

    df = load_csv(dataset_path)
    profile = profile_data(df)

    print("=== PROFILO DEL DATASET ===")
    print(f"\nNumero di righe: {profile['rows']}")
    print(f"Numero di colonne: {profile['columns']}")

    print("\nValori mancanti:")
    for column, count in profile["missing_values"].items():
        print(f"- {column}: {count}")

    print("\n=== ANALISI DELLE VENDITE ===")

    total = calculate_total_revenue(df)

    print(f"\nRicavi totali: {total:.2f}")

    product_summary = revenue_by_product(df)

    print("\nRicavi per prodotto:")
    print(product_summary.to_string(index=False))

    top_product = product_summary.iloc[0]

    print(
        f"\nProdotto con i ricavi più alti: "
        f"{top_product['product']} "
        f"({top_product['revenue']:.2f})"
    )

    monthly_summary = revenue_by_month(df)

    print("\nRicavi per mese:")
    print(monthly_summary.to_string(index=False))
    
    chart_path = plot_revenue_by_month(monthly_summary)

    print(f"\nGrafico salvato in: {chart_path}")