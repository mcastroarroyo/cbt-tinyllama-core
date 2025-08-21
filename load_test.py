import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the model ID from the Hugging Face Hub
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

print(f"--- Loading Model: {model_id} ---")
print("Note: The first time you run this, it will download the model (approx. 16 GB), which can take a while.")

# 1. Load the Tokenizer
# The tokenizer converts text into a format the model understands.
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Load the Model
# This will download the model weights and load them into memory.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto", # Automatically uses GPU/MPS if available, otherwise CPU
)

print("--- Model Loaded Successfully ---")

# 3. Create a Prompt
# The Instruct version of Llama 3 uses a specific chat template.
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Explain the concept of self-attention in a transformer model in one paragraph."},
]

# Apply the chat template to the messages
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 4. Run Inference
print("\n--- Running Inference ---")
# Convert the prompt text to input tensors
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate the response from the model
outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

# 5. Decode and print the response
response = outputs[0][inputs.input_ids.shape[-1]:]
print("\nModel Response:")
print(tokenizer.decode(response, skip_special_tokens=True))
