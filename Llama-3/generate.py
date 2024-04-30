import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from peft import prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    GenerationConfig
)
from trl import SFTTrainer


model_name = "meta-llama/Meta-Llama-3-8B"
#Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#Create a new token and add it to the tokenizer
tokenizer.add_special_tokens({"pad_token":"<pad>"})
tokenizer.padding_side = 'left'

dataset = load_dataset("json", data_files="../huggingface-lyrics-data.json", split="train")
dataset = dataset.train_test_split(test_size=0.1)

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
          model_name, quantization_config=bnb_config, device_map={"": 0}
)

#Resize the embeddings
model.resize_token_embeddings(len(tokenizer))
#Configure the pad token in the model
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False # Gradient checkpointing is used by default but not compatible with caching

model = PeftModel.from_pretrained(model, "./results/checkpoint-10")

def generate(instruction):
    prompt = "### Human: "+instruction+"### Assistant: "
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model.generate(
            input_ids=input_ids,
            generation_config=GenerationConfig(temperature=1.0, top_p=1.0, top_k=50, num_beams=1),
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256
    )
    for seq in generation_output.sequences:
        output = tokenizer.decode(seq)
        print(output.split("### Assistant: ")[1].strip())
        
generate("Write a song in the style of Taylor Swift")