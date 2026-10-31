import json

from agent import choose_tool, choose_tool_plan


SINGLE_TOOL_CASES = [
    {
        "question": "Quanto sono i ricavi totali?",
        "expected": {
            "tool": "total_revenue",
            "arguments": {},
        },
    },
    {
        "question": "Quale prodotto ha generato più ricavi?",
        "expected": {
            "tool": "product_revenue",
            "arguments": {},
        },
    },
    {
        "question": "Quali ricavi ha generato il Laptop?",
        "expected": {
            "tool": "product_revenue",
            "arguments": {"product": "Laptop"},
        },
    },
    {
        "question": "Mostrami i ricavi per mese.",
        "expected": {
            "tool": "monthly_revenue",
            "arguments": {},
        },
    },
    {
        "question": "Crea un grafico dei ricavi mensili.",
        "expected": {
            "tool": "monthly_chart",
            "arguments": {},
        },
    },
    {
        "question": "Ci sono valori mancanti nel dataset?",
        "expected": {
            "tool": "dataset_profile",
            "arguments": {},
        },
    },
]


def evaluate_single_tools() -> tuple[int, int]:
    """Valuta la selezione dei singoli strumenti."""

    passed = 0
    total = len(SINGLE_TOOL_CASES)

    print("\n=== VALUTAZIONE DEI SINGOLI STRUMENTI ===")

    for case in SINGLE_TOOL_CASES:
        question = case["question"]
        expected = case["expected"]

        actual = choose_tool(question)

        success = actual == expected

        print(
            f"\n[{'OK' if success else 'ERRORE'}] "
            f"{question}"
        )

        print(
            "Atteso:  "
            f"{json.dumps(expected, ensure_ascii=False)}"
        )

        print(
            "Ottenuto: "
            f"{json.dumps(actual, ensure_ascii=False)}"
        )

        if success:
            passed += 1

    return passed, total


def evaluate_workflow() -> tuple[int, int]:
    """Valuta la scelta e il piano del workflow multi-step."""

    question = "Analizza i ricavi per prodotto e per mese."

    print("\n=== VALUTAZIONE DEL WORKFLOW ===")

    passed = 0
    total = 2

    expected_tool = {
        "tool": "sales_overview",
        "arguments": {},
    }

    actual_tool = choose_tool(question)

    tool_success = actual_tool == expected_tool

    print(
        f"\n[{'OK' if tool_success else 'ERRORE'}] "
        "Selezione di sales_overview"
    )

    print(f"Atteso:  {expected_tool}")
    print(f"Ottenuto: {actual_tool}")

    if tool_success:
        passed += 1

    expected_plan = [
        {
            "tool": "product_revenue",
            "arguments": {},
        },
        {
            "tool": "monthly_revenue",
            "arguments": {},
        },
    ]

    actual_plan = choose_tool_plan(question)

    plan_success = actual_plan == expected_plan

    print(
        f"\n[{'OK' if plan_success else 'ERRORE'}] "
        "Piano multi-step"
    )

    print(f"Atteso:  {expected_plan}")
    print(f"Ottenuto: {actual_plan}")

    if plan_success:
        passed += 1

    return passed, total


def main() -> None:
    total_passed = 0
    total_tests = 0

    passed, total = evaluate_single_tools()
    total_passed += passed
    total_tests += total

    passed, total = evaluate_workflow()
    total_passed += passed
    total_tests += total

    accuracy = total_passed / total_tests * 100

    print("\n=== RISULTATO FINALE ===")
    print(f"Verifiche superate: {total_passed}/{total_tests}")
    print(f"Accuratezza: {accuracy:.1f}%")

    if total_passed < total_tests:
        print(
            "Alcune verifiche non sono state superate. "
            "Analizza i risultati prima di modificare il codice."
        )


if __name__ == "__main__":
    main()