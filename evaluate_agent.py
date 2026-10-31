import json

from agent import (
    choose_tool,
    choose_tool_plan,
    execute_tool,
    execute_tool_plan,
)

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

OUTPUT_CASES = [
    {
        "name": "Ricavi totali",
        "tool": "total_revenue",
        "expected": ["6111.00"],
    },
    {
        "name": "Ricavi per prodotto",
        "tool": "product_revenue",
        "expected": [
            "Laptop",
            "3650",
            "Monitor",
            "1620",
            "Keyboard",
            "465",
            "Mouse",
            "376",
        ],
    },
    {
        "name": "Ricavi mensili",
        "tool": "monthly_revenue",
        "expected": [
            "2026-01",
            "2540",
            "2026-02",
            "1875",
            "2026-03",
            "1696",
        ],
    },
    {
        "name": "Profilo del dataset",
        "tool": "dataset_profile",
        "expected": [
            "Righe: 10",
            "Colonne: 5",
            "region: 1",
        ],
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

def evaluate_tool_outputs() -> tuple[int, int]:
    """
    Verifica che gli strumenti restituiscano
    i risultati attesi dal dataset.
    """

    passed = 0
    total = len(OUTPUT_CASES) + 1

    print("\n=== VALUTAZIONE DEI RISULTATI ===")

    for case in OUTPUT_CASES:
        name = case["name"]
        tool = case["tool"]
        expected = case["expected"]

        try:
            result = execute_tool(tool)

            missing = [
                value
                for value in expected
                if value not in result
            ]

            success = len(missing) == 0

            print(
                f"\n[{'OK' if success else 'ERRORE'}] {name}"
            )

            if not success:
                print(f"Valori mancanti: {missing}")

            if success:
                passed += 1

        except Exception as exc:
            print(f"\n[ERRORE] {name}: {exc}")

    # Verifichiamo anche il risultato del workflow multi-step.
    print("\n=== VALUTAZIONE WORKFLOW MULTI-STEP ===")

    plan = [
        {
            "tool": "product_revenue",
            "arguments": {},
        },
        {
            "tool": "monthly_revenue",
            "arguments": {},
        },
    ]

    expected_workflow = [
        "Laptop",
        "3650",
        "Monitor",
        "1620",
        "2026-01",
        "2540",
        "2026-02",
        "1875",
        "2026-03",
        "1696",
    ]

    try:
        result = execute_tool_plan(plan)

        missing = [
            value
            for value in expected_workflow
            if value not in result
        ]

        success = len(missing) == 0

        print(
            f"\n[{'OK' if success else 'ERRORE'}] "
            "Completezza del risultato multi-step"
        )

        if not success:
            print(f"Valori mancanti: {missing}")

        if success:
            passed += 1

    except Exception as exc:
        print(f"\n[ERRORE] Workflow: {exc}")

    return passed, total

def main() -> None:
    routing_passed = 0
    routing_total = 0

    passed, total = evaluate_single_tools()
    routing_passed += passed
    routing_total += total

    passed, total = evaluate_workflow()
    routing_passed += passed
    routing_total += total

    output_passed, output_total = evaluate_tool_outputs()

    print("\n=== RISULTATO FINALE ===")

    routing_accuracy = routing_passed / routing_total * 100
    output_accuracy = output_passed / output_total * 100

    print(
        f"Routing e pianificazione: "
        f"{routing_passed}/{routing_total} "
        f"({routing_accuracy:.1f}%)"
    )

    print(
        f"Qualità dei risultati: "
        f"{output_passed}/{output_total} "
        f"({output_accuracy:.1f}%)"
    )


if __name__ == "__main__":
    main()