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
    prompt = "Tell me about AI"
    prompt_template=f'''<s>[INST] {prompt} [/INST]
    '''
    
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, 
                            temperature=0.7, 
                            do_sample=True, 
                            top_p=0.95, 
                            top_k=40, 
                            max_new_tokens=512)

    return tokenizer.decode(output[0], skip_special_tokens=True)
