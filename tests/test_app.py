import unittest
from unittest.mock import patch

from app import answer_question


class TestApp(unittest.TestCase):

    def test_missing_upload(self):
        """L'interfaccia rifiuta una richiesta senza file."""

        with patch("app.run_agent") as mock_agent:
            result, chart = answer_question(
                None,
                "Quanto sono i ricavi totali?",
            )

        self.assertEqual(
            result,
            "Carica prima un file CSV o Excel.",
        )
        self.assertIsNone(chart)
        mock_agent.assert_not_called()

    def test_empty_question(self):
        """L'interfaccia rifiuta una domanda vuota."""

        with patch("app.run_agent") as mock_agent:
            result, chart = answer_question(
                "sales.csv",
                "   ",
            )

        self.assertEqual(result, "Inserisci una domanda.")
        self.assertIsNone(chart)
        mock_agent.assert_not_called()

    def test_invalid_dataset(self):
        """L'interfaccia mostra un errore comprensibile per dati non validi."""

        error_message = (
            "Colonne necessarie mancanti: ['unit_price']"
        )

        with patch(
            "app.run_agent",
            side_effect=ValueError(error_message),
        ):
            result, chart = answer_question(
                "test_invalid.csv",
                "Quanto sono i ricavi totali?",
            )

        self.assertEqual(
            result,
            f"Non riesco ad analizzare il dataset: {error_message}",
        )
        self.assertIsNone(chart)

    def test_missing_file(self):
        """L'interfaccia gestisce un file non più disponibile."""

        with patch(
            "app.run_agent",
            side_effect=FileNotFoundError("File non trovato"),
        ):
            result, chart = answer_question(
                "missing.csv",
                "Quanto sono i ricavi totali?",
            )

        self.assertEqual(
            result,
            "Il file caricato non è più disponibile. "
            "Caricalo nuovamente.",
        )
        self.assertIsNone(chart)

    def test_file_read_error(self):
        """L'interfaccia gestisce un errore di lettura."""

        with patch(
            "app.run_agent",
            side_effect=OSError("Errore simulato"),
        ):
            with self.assertLogs(level="ERROR"):
                result, chart = answer_question(
                    "sales.csv",
                    "Quanto sono i ricavi totali?",
                )

        self.assertEqual(
            result,
            "Non riesco a leggere il file. "
            "Verifica che sia un CSV o Excel valido e non danneggiato.",
        )
        self.assertIsNone(chart)

    def test_unexpected_error(self):
        """Gli errori imprevisti non interrompono la funzione."""

        with patch(
            "app.run_agent",
            side_effect=RuntimeError("Errore simulato"),
        ):
            with self.assertLogs(level="ERROR"):
                result, chart = answer_question(
                    "sales.csv",
                    "Quanto sono i ricavi totali?",
                )

        self.assertEqual(
            result,
            "Si è verificato un errore imprevisto durante l'analisi. "
            "Controlla il terminale per maggiori dettagli.",
        )
        self.assertIsNone(chart)

    def test_successful_answer(self):
        """L'interfaccia restituisce correttamente una risposta valida."""

        expected = (
            "Strumento selezionato: total_revenue\n\n"
            "Ricavi totali: 6111.00"
        )

        with patch("app.run_agent", return_value=expected) as mock_agent:
            result, chart = answer_question(
                "sales.csv",
                "Quanto sono i ricavi totali?",
            )

        self.assertEqual(result, expected)
        self.assertIsNone(chart)

        mock_agent.assert_called_once_with(
            "Quanto sono i ricavi totali?",
            dataset_path="sales.csv",
        )


if __name__ == "__main__":
    unittest.main()