import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer
from modeling_llama import LlamaForCausalLM # Import your custom model
from datasets import load_dataset
import torch.nn.functional as F

# --- 1. Define the Disentanglement Loss Function ---
def cosine_similarity_loss(attentions):
    """
    Calculates a loss that penalizes similarity between attention heads.
    """
    # attentions is a tuple of attention matrices, one for each layer
    # Let's focus on the attentions of a single layer for simplicity, e.g., the last layer
    last_layer_attentions = attentions[-1] # Shape: (batch_size, num_heads, seq_len, seq_len)
    
    batch_size, num_heads, seq_len, _ = last_layer_attentions.shape
    
    # Flatten the attention maps for comparison
    flat_attentions = last_layer_attentions.view(batch_size, num_heads, -1)
    
    # Normalize for cosine similarity calculation
    norm_attentions = F.normalize(flat_attentions, p=2, dim=2)
    
    # Calculate pairwise cosine similarity between all heads
    # (batch_size, num_heads, seq_len*seq_len) x (batch_size, seq_len*seq_len, num_heads) -> (batch_size, num_heads, num_heads)
    similarity_matrix = torch.matmul(norm_attentions, norm_attentions.transpose(1, 2))
    
    # We want to minimize the similarity between different heads.
    # The ideal matrix is an identity matrix (1s on diagonal, 0s off-diagonal).
    # We can create a loss that penalizes the off-diagonal elements.
    identity = torch.eye(num_heads, device=similarity_matrix.device).unsqueeze(0).expand(batch_size, -1, -1)
    
    # Penalize the absolute difference from the identity matrix
    loss = torch.mean(torch.abs(similarity_matrix - identity))
    
    return loss

# --- 2. Create a Custom Trainer Class ---
class CBTTrainer(Trainer):
    def __init__(self, disentangle_lambda=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disentangle_lambda = disentangle_lambda

    def compute_loss(self, model, inputs, return_outputs=False):
        # Tell the model we want the attention weights
        outputs = model(**inputs, output_attentions=True)
        
        # Standard language modeling loss
        lm_loss = outputs.loss
        
        # Custom disentanglement loss
        attentions = outputs.attentions
        disentangle_loss = cosine_similarity_loss(attentions)
        
        # Combine the losses
        total_loss = lm_loss + self.disentangle_lambda * disentangle_loss
        
        return (total_loss, outputs) if return_outputs else total_loss

# --- 3. Main Training Setup ---
if __name__ == "__main__":
    from transformers import BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig

    # --- Model and Tokenizer ---
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- QLoRA Configuration ---
    # This loads the model in 4-bit precision
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("Loading custom 4-bit model...")
    # This time, we use both quantization_config and device_map
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
    )

    # --- PEFT (LoRA) Configuration ---
    # This freezes the base model and adds small, trainable adapter layers
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Dataset ---
    print("Loading and preparing dataset...")
    dataset = load_dataset("Abirate/english_quotes", split="train") 
    
    def tokenize_function(examples):
        return tokenizer(examples["quote"], truncation=True, padding="max_length", max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("input_ids", "labels")

    # --- Training ---
    training_args = TrainingArguments(
        output_dir="./cbt_results",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,  # Saves more memory
        save_steps=500,
        logging_steps=50,
        learning_rate=2e-4, # Higher learning rate is common for LoRA
        fp16=False,
    )

    trainer = CBTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        disentangle_lambda=0.1
    )

    print("Starting QLoRA training...")
    trainer.train()
    print("Training complete.")