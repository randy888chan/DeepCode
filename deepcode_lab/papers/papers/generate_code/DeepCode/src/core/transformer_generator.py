import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class TransformerModel:
    def __init__(self, model_name='facebook/bart-base'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_code(self, prompt, max_length=512, temperature=0.7, num_beams=5):
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model.generate(
            inputs['input_ids'],
            max_length=max_length,
            temperature=temperature,
            num_beams=num_beams,
            early_stopping=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    model = TransformerModel()
    generated_code = model.generate_code("def hello_world():\n    print(\"Hello, world!\")")
    print(generated_code)