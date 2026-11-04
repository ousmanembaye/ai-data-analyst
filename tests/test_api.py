import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app

import os

from io import BytesIO
from openpyxl import Workbook

client = TestClient(app)


class TestAPI(unittest.TestCase):

    def test_health_endpoint(self):
        """Verifica che l'endpoint health funzioni."""

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)

        self.assertEqual(
            response.json(),
            {
                "status": "ok",
                "message": "AI Data Analyst API is running",
            },
        )

    @patch("api.run_agent")
    def test_analyze_success(self, mock_agent):
        """Verifica una richiesta POST valida."""

        mock_agent.return_value = "Ricavi totali: 6111.00"

        response = client.post(
            "/analyze",
            json={
                "question": "Quanto sono i ricavi totali?"
            },
        )

        self.assertEqual(response.status_code, 200)

        data = response.json()

        self.assertEqual(data["status"], "success")
        self.assertEqual(
            data["question"],
            "Quanto sono i ricavi totali?",
        )
        self.assertEqual(
            data["answer"],
            "Ricavi totali: 6111.00",
        )

        mock_agent.assert_called_once_with(
            "Quanto sono i ricavi totali?"
        )

    @patch("api.run_agent")
    def test_analyze_empty_question(self, mock_agent):
        """Verifica che una domanda vuota venga rifiutata."""

        response = client.post(
            "/analyze",
            json={"question": ""},
        )

        self.assertEqual(response.status_code, 422)

        mock_agent.assert_not_called()

    @patch("api.run_agent")
    def test_analyze_wrong_type(self, mock_agent):
        """Verifica che un numero non venga accettato come domanda."""

        response = client.post(
            "/analyze",
            json={"question": 123},
        )

        self.assertEqual(response.status_code, 422)

        mock_agent.assert_not_called()

    @patch("api.run_agent")
    def test_analyze_missing_question(self, mock_agent):
        """Verifica che venga rifiutata una richiesta senza question."""

        response = client.post(
            "/analyze",
            json={},
        )

        self.assertEqual(response.status_code, 422)

        mock_agent.assert_not_called()

    @patch("api.run_agent")
    def test_agent_value_error(self, mock_agent):
        """Verifica la gestione di un errore ValueError."""

        mock_agent.side_effect = ValueError("Dataset non valido")

        response = client.post(
            "/analyze",
            json={"question": "Analizza i dati"},
        )

        self.assertEqual(response.status_code, 400)

        self.assertEqual(
            response.json()["detail"],
            "Dataset non valido",
        )

        mock_agent.assert_called_once()

    def test_analyze_uploaded_csv(self):
        csv_content = (
            b"date,product,quantity,unit_price,region\n"
            b"2026-01-01,Laptop,1,100,North\n"
        )

        def fake_run_agent(question, dataset_path):
            self.assertEqual(
                question,
                "Quale prodotto ha generato più ricavi?"
            )

            # Il file deve esistere durante l'analisi.
            self.assertTrue(os.path.exists(dataset_path))

            # Verifichiamo che il contenuto sia corretto.
            with open(dataset_path, "rb") as uploaded_file:
                self.assertEqual(uploaded_file.read(), csv_content)

            return "Prodotto più redditizio: Laptop"

        with patch("api.run_agent", side_effect=fake_run_agent) as mock_agent:
            response = client.post(
                "/analyze-upload",
                data={
                    "question": "Quale prodotto ha generato più ricavi?"
                },
                files={
                    "file": ("sales.csv", csv_content, "text/csv")
                }
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["status"],
            "success"
        )
        self.assertIn("Laptop", response.json()["answer"])

        # Verifichiamo che il file temporaneo sia stato eliminato.
        temporary_path = mock_agent.call_args.kwargs["dataset_path"]
        self.assertFalse(os.path.exists(temporary_path))

    def test_analyze_unsupported_extension(self):
        with patch("api.run_agent") as mock_agent:
            response = client.post(
                "/analyze-upload",
                data={"question": "Analizza questo file"},
                files={
                    "file": ("sales.txt", b"contenuto", "text/plain")
                }
            )

        self.assertEqual(response.status_code, 400)

        self.assertEqual(
            response.json()["detail"],
            "Formato non supportato. Carica un file CSV o XLSX."
        )

        mock_agent.assert_not_called()

    def test_analyze_uploaded_xlsx(self):
        workbook = Workbook()
        sheet = workbook.active

        sheet.append(["date", "product", "quantity", "unit_price", "region"])
        sheet.append(["2026-01-01", "Laptop", 1, 100, "North"])

        excel_buffer = BytesIO()
        workbook.save(excel_buffer)

        xlsx_content = excel_buffer.getvalue()

        def fake_run_agent(question, dataset_path):
            self.assertTrue(os.path.exists(dataset_path))
            self.assertTrue(dataset_path.endswith(".xlsx"))

            with open(dataset_path, "rb") as uploaded_file:
                self.assertEqual(uploaded_file.read(), xlsx_content)

            return "Analisi Excel completata"

        with patch("api.run_agent", side_effect=fake_run_agent) as mock_agent:
            response = client.post(
                "/analyze-upload",
                data={"question": "Analizza il file Excel"},
                files={
                    "file": (
                        "sales.xlsx",
                        xlsx_content,
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                }
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        self.assertEqual(response.json()["answer"], "Analisi Excel completata")

        temporary_path = mock_agent.call_args.kwargs["dataset_path"]
        self.assertFalse(os.path.exists(temporary_path))

    def test_analyze_file_too_large(self):
        large_content = b"x" * (10 * 1024 * 1024 + 1)

        with patch("api.run_agent") as mock_agent:
            response = client.post(
                "/analyze-upload",
                data={"question": "Analizza questo file"},
                files={
                    "file": ("large.csv", large_content, "text/csv")
                }
            )

        self.assertEqual(response.status_code, 413)

        self.assertEqual(
            response.json()["detail"],
            "File troppo grande. Dimensione massima consentita: 10 MB."
        )

        mock_agent.assert_not_called()

if __name__ == "__main__":
    unittest.main()