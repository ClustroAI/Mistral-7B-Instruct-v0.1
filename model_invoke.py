import json
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

def invoke(input_text):
    #try:
    #    input_json = json.loads(input_text)
    #except:

    # messages = [
    #     {"role": "user", "content": "What is your favourite condiment?"},
    #     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    #     {"role": "user", "content": "Do you have mayonnaise recipes?"}
    # ]

    # encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    # model_inputs = encodeds.to(device)
    # model.to(device)

    # generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    # decoded = tokenizer.batch_decode(generated_ids)
    # return decoded[0]
  
    input_json = json.loads(input_text)
    prompt = input_json['prompt']
    prompt_template=f'''<s>[INST] {prompt} [/INST]
    '''
    
    # Generator configs
    temperature = input_json['temperature'] if 'temperature' in input_json else 0.7
    top_p = input_json['top_p'] if 'top_p' in input_json else 0.95
    top_k = input_json['top_k'] if 'top_k' in input_json else 40
    max_new_tokens = input_json['max_new_tokens'] if 'max_new_tokens' in input_json else 512
  
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, 
                            temperature=temperature, 
                            do_sample=True, 
                            top_p=top_p, 
                            top_k=top_k, 
                            max_new_tokens=max_new_tokens)

    return tokenizer.decode(output[0, input_ids.shape[1]:-1], skip_special_tokens=True)
