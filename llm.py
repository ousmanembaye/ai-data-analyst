import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inizialmente non carichiamo il modello.
# Verrà caricato soltanto quando servirà.
_tokenizer = None
_model = None


def load_model():
    """
    Carica il tokenizer e il modello una sola volta.
    """

    global _tokenizer, _model

    if _model is None or _tokenizer is None:
        print(f"Caricamento del modello: {MODEL_NAME}")
        print(f"Dispositivo utilizzato: {DEVICE}")

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

        _model.to(DEVICE)
        _model.eval()

        print("Modello caricato correttamente.")

    return _tokenizer, _model


def generate_reply(
    user_message: str,
    system_prompt: str = (
        "Sei un assistente AI. "
        "Rispondi in italiano in modo chiaro e conciso."
    ),
) -> str:
    """
    Genera una risposta utilizzando il modello linguistico.
    """

    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError("Il messaggio non può essere vuoto.")

    tokenizer, model = load_model()

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    # Applichiamo il formato conversazionale del modello.
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Convertiamo il testo in token e li trasferiamo sul dispositivo.
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
    ).to(DEVICE)

    # Generiamo la risposta senza calcolare gradienti.
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Estraiamo soltanto i token generati dal modello,
    # escludendo quelli che contengono la domanda.
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


if __name__ == "__main__":
    question = "Spiega in una frase che cos'è un file CSV."

    answer = generate_reply(question)

    print("\n=== RISPOSTA DEL MODELLO ===")
    print(answer)