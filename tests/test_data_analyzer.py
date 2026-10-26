from pathlib import Path
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from data_analyzer import (
    load_csv,
    load_dataset,
    profile_data,
    add_revenue_column,
    calculate_total_revenue,
    revenue_by_product,
    revenue_by_month,
    plot_revenue_by_month,
)


class TestDataAnalyzer(unittest.TestCase):

    def setUp(self):
        """Prepara un dataset piccolo e ripetibile per ogni test."""

        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        data = pd.DataFrame({
            "date": [
                "2026-01-05",
                "2026-01-10",
                "2026-02-02",
                "2026-03-01",
            ],
            "product": [
                "Laptop",
                "Mouse",
                "Laptop",
                "Mouse",
            ],
            "quantity": [2, 10, 1, 8],
            "unit_price": [900, 20, 950, 22],
            "region": ["Nord", "Nord", "Sud", None],
        })

        self.csv_path = self.temp_path / "sales.csv"
        data.to_csv(self.csv_path, index=False)

        self.df = load_csv(self.csv_path)

    def tearDown(self):
        """Elimina i file temporanei al termine di ogni test."""

        self.temp_dir.cleanup()

    def test_load_csv(self):
        self.assertEqual(len(self.df), 4)
        self.assertEqual(len(self.df.columns), 5)

    def test_missing_file(self):
        missing_path = self.temp_path / "missing.csv"

        with self.assertRaises(FileNotFoundError):
            load_csv(missing_path)

    def test_profile_data(self):
        profile = profile_data(self.df)

        self.assertEqual(profile["rows"], 4)
        self.assertEqual(profile["columns"], 5)
        self.assertEqual(profile["missing_values"]["region"], 1)

    def test_add_revenue_column(self):
        result = add_revenue_column(self.df)

        self.assertIn("revenue", result.columns)
        self.assertEqual(
            result["revenue"].tolist(),
            [1800, 200, 950, 176],
        )

        # Il DataFrame originale non deve essere modificato.
        self.assertNotIn("revenue", self.df.columns)

    def test_calculate_total_revenue(self):
        total = calculate_total_revenue(self.df)

        self.assertEqual(total, 3126)

    def test_revenue_by_product(self):
        summary = revenue_by_product(self.df)

        result = dict(zip(summary["product"], summary["revenue"]))

        self.assertEqual(result["Laptop"], 2750)
        self.assertEqual(result["Mouse"], 376)

        # I prodotti devono essere ordinati per ricavo decrescente.
        self.assertEqual(
            summary["product"].tolist(),
            ["Laptop", "Mouse"],
        )

    def test_revenue_by_month(self):
        summary = revenue_by_month(self.df)

        self.assertEqual(
            summary["month"].tolist(),
            ["2026-01", "2026-02", "2026-03"],
        )

        self.assertEqual(
            summary["revenue"].tolist(),
            [2000, 950, 176],
        )

    def test_missing_required_column(self):
        incomplete_df = self.df.drop(columns=["unit_price"])

        with self.assertRaises(ValueError):
            add_revenue_column(incomplete_df)

    def test_plot_revenue_by_month(self):
        monthly_summary = revenue_by_month(self.df)
        chart_dir = self.temp_path / "charts"

        chart_path = plot_revenue_by_month(
            monthly_summary,
            output_dir=chart_dir,
        )

        self.assertTrue(chart_path.is_file())
        self.assertGreater(chart_path.stat().st_size, 0)

    def test_load_dataset_csv(self):
        result = load_dataset(self.csv_path)

        self.assertEqual(len(result), 4)
        self.assertEqual(len(result.columns), 5)

    def test_load_dataset_xlsx(self):
        xlsx_path = self.temp_path / "sales.xlsx"

        self.df.to_excel(xlsx_path, index=False)

        result = load_dataset(xlsx_path)

        self.assertEqual(len(result), 4)
        self.assertEqual(len(result.columns), 5)

    def test_load_dataset_unsupported_extension(self):
        txt_path = self.temp_path / "sales.txt"
        txt_path.write_text("test", encoding="utf-8")

        with self.assertRaises(ValueError):
            load_dataset(txt_path)

    def test_add_revenue_column_invalid_quantity(self):
        """Rifiuta quantità che non sono numeriche."""

        invalid_df = self.df.copy()
        invalid_df["quantity"] = ["abc", 10, 1, 8]

        with self.assertRaisesRegex(ValueError, "quantity"):
            add_revenue_column(invalid_df)

    def test_add_revenue_column_missing_product_value(self):
        """Rifiuta righe senza il nome del prodotto."""

        invalid_df = self.df.copy()
        invalid_df.loc[0, "product"] = None

        with self.assertRaisesRegex(ValueError, "product"):
            add_revenue_column(invalid_df)

    def test_revenue_by_month_missing_date_column(self):
        """Rifiuta l'analisi mensile se manca la colonna date."""

        invalid_df = self.df.drop(columns=["date"])

        with self.assertRaisesRegex(ValueError, "date"):
            revenue_by_month(invalid_df)

    def test_revenue_by_month_invalid_date(self):
        """Rifiuta date che non possono essere interpretate."""

        invalid_df = self.df.copy()
        invalid_df.loc[0, "date"] = "data-sbagliata"

        with self.assertRaisesRegex(ValueError, "non valide"):
            revenue_by_month(invalid_df)

    def test_load_csv_header_without_rows(self):
        """Rifiuta un CSV che contiene le intestazioni ma nessun dato."""

        empty_csv = self.temp_path / "empty.csv"
        empty_csv.write_text(
            "date,product,quantity,unit_price,region\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "righe di dati"):
            load_csv(empty_csv)


if __name__ == "__main__":
    unittest.main()