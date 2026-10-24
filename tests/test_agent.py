import unittest
from pathlib import Path
from unittest.mock import patch

from agent import (
    normalize_text,
    fallback_routing,
    choose_tool,
    execute_tool,
    run_agent,
)


class TestAgentRouting(unittest.TestCase):

    def test_normalize_text(self):
        result = normalize_text("È POSSIBILE analizzare le vendite?")
        self.assertEqual(result, "e possibile analizzare le vendite?")

    def test_fallback_chart(self):
        result = fallback_routing("Crea un grafico delle vendite")
        self.assertEqual(result, "monthly_chart")

    def test_fallback_monthly_revenue(self):
        result = fallback_routing("Mostrami i ricavi per mese")
        self.assertEqual(result, "monthly_revenue")

    def test_fallback_product_revenue(self):
        result = fallback_routing(
            "Quale prodotto ha generato più ricavi?"
        )
        self.assertEqual(result, "product_revenue")

    def test_fallback_dataset_profile(self):
        result = fallback_routing(
            "Ci sono valori mancanti nel dataset?"
        )
        self.assertEqual(result, "dataset_profile")

    def test_fallback_total_revenue(self):
        result = fallback_routing("Qual è il fatturato totale?")
        self.assertEqual(result, "total_revenue")

    def test_fallback_unknown_request(self):
        result = fallback_routing("Ciao, come stai?")
        self.assertIsNone(result)

    def test_fallback_paraphrases(self):
        cases = [
            (
                "Qual è l'articolo più redditizio?",
                "product_revenue",
            ),
            (
                "Quale articolo porta più soldi?",
                "product_revenue",
            ),
            (
                "Fammi una classifica dei prodotti per fatturato",
                "product_revenue",
            ),
            (
                "Quanto abbiamo incassato in totale?",
                "total_revenue",
            ),
            (
                "Come sono andate le vendite nei tre mesi?",
                "monthly_revenue",
            ),
            (
                "Quanti campi ha la tabella?",
                "dataset_profile",
            ),
            (
                "Genera un diagramma delle vendite",
                "monthly_chart",
            ),
        ]

        for question, expected_tool in cases:
            with self.subTest(question=question):
                self.assertEqual(
                    fallback_routing(question),
                    expected_tool,
                )

    @patch("agent.generate_reply")
    def test_choose_tool_accepts_valid_tool(self, mock_generate):
        mock_generate.return_value = "product_revenue"

        result = choose_tool("Quale prodotto ha generato più ricavi?")

        self.assertEqual(result, "product_revenue")
        mock_generate.assert_called_once()

    @patch("agent.generate_reply")
    def test_choose_tool_uses_fallback_for_invalid_output(
        self, mock_generate
    ):
        mock_generate.return_value = "Non so quale strumento scegliere."

        result = choose_tool("Quale prodotto ha generato più ricavi?")

        self.assertEqual(result, "product_revenue")

    @patch("agent.generate_reply")
    def test_choose_tool_returns_none_for_unknown_request(
        self, mock_generate
    ):
        mock_generate.return_value = "Risposta non valida"

        result = choose_tool("Ciao, come stai?")

        self.assertIsNone(result)


class TestAgentTools(unittest.TestCase):

    def test_total_revenue(self):
        result = execute_tool("total_revenue")

        self.assertIn("6111.00", result)

    def test_product_revenue(self):
        result = execute_tool("product_revenue")

        self.assertIn("Laptop", result)
        self.assertIn("3650", result)

    def test_monthly_revenue(self):
        result = execute_tool("monthly_revenue")

        self.assertIn("2026-01", result)
        self.assertIn("2026-02", result)
        self.assertIn("2026-03", result)

    def test_dataset_profile(self):
        result = execute_tool("dataset_profile")

        self.assertIn("Righe: 10", result)
        self.assertIn("Colonne: 5", result)
        self.assertIn("region: 1", result)

    @patch("agent.plot_revenue_by_month")
    def test_monthly_chart(self, mock_plot):
        mock_plot.return_value = Path("fake_chart.png")

        result = execute_tool("monthly_chart")

        self.assertIn("fake_chart.png", result)
        mock_plot.assert_called_once()

    def test_unauthorized_tool(self):
        with self.assertRaises(ValueError):
            execute_tool("execute_arbitrary_code")

    @patch("agent.choose_tool")
    def test_run_agent_with_empty_input(self, mock_choose):
        result = run_agent("   ")

        self.assertEqual(result, "Inserisci una domanda.")
        mock_choose.assert_not_called()

    @patch("agent.choose_tool")
    def test_run_agent_total_revenue(self, mock_choose):
        mock_choose.return_value = "total_revenue"

        result = run_agent("Quanto sono i ricavi totali?")

        self.assertIn("6111.00", result)
        mock_choose.assert_called_once()


if __name__ == "__main__":
    unittest.main()