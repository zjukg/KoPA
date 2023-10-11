import os
import json
import torch
import transformers
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

base_path = 'YOUR LLM PATH'

prompt_template = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Given a triple from a knowledge graph. Each triple consists of a head entity, a relation, and a tail entity. Please determine the correctness of the triple and response True or False.

### Input:
{}

### Response:

"""

def load_test_dataset(path):
    test_dataset = json.load(open(path, "r"))
    return test_dataset

if __name__ == "__main__":
    cuda = "cuda:0"
    lora_weights = "YOUR SAVE PATH"
    test_data_path = "data/UMLS-test.json"
    embedding_path = "{}/embeddings.pth".format(lora_weights)
    test_dataset = load_test_dataset(test_data_path)
    kg_embeddings = torch.load(embedding_path).to(cuda)
    tokenizer = LlamaTokenizer.from_pretrained(base_path)
    model = LlamaForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16
    ).to(cuda)
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    ).to(cuda)
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model = model.eval()
    result = []
    for data in test_dataset:
        ent = data["input"]
        ans = data["output"]
        ids = data["embedding_ids"]
        ids = torch.LongTensor(ids).reshape(1, -1).to(cuda)
        prefix = kg_embeddings(ids)
        prompt = prompt_template.format(ent)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(cuda)
        token_embeds = model.model.model.embed_tokens(input_ids)
        input_embeds = torch.cat((prefix, token_embeds), dim=1)
        generate_ids = model.generate(
            inputs_embeds=input_embeds, 
            max_new_tokens=16
        )
        context = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = response.replace(context, "").strip()
        print(response + '\n')
        result.append(
            {
                "answer": ans,
                "predict": response
            }
        )
    answer = []
    predict = []
    for data in result:
        if "True" in data["answer"]:
            answer.append(1)
        else:
            answer.append(0)
        if "True" in data["predict"]:
            predict.append(1)
        else:
            predict.append(0)
    acc = accuracy_score(y_true=answer, y_pred=predict)
    p = precision_score(y_true=answer, y_pred=predict)
    r = recall_score(y_true=answer, y_pred=predict)
    f1 = f1_score(y_true=answer, y_pred=predict)
    print(acc, p, r, f1)
    

    
