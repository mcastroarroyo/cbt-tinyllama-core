import torch
from transformers import AutoTokenizer
# This line now correctly imports the LlamaForCausalLM class from your local file
from modeling_llama import LlamaForCausalLM

# Define the model ID from the Hugging Face Hub
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

print(f"--- Loading Model: {model_id} ---")

# 1. Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Load the Model
# This line now correctly uses the LlamaForCausalLM class
model = LlamaForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto", # Automatically uses GPU/MPS if available, otherwise CPU
)

print("--- Model Loaded Successfully ---")

# 3. Create a Prompt
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
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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
