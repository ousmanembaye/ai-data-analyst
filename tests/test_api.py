import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app


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


if __name__ == "__main__":
    unittest.main()