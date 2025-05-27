import os
import torch 
from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import ast




os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"


model_id = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoAWQForCausalLM.from_pretrained(
  model_id,
  torch_dtype=torch.float16,
  low_cpu_mem_usage=True,
  device_map="auto",
)




def get_response(
    user_prompt, max_tokens=500, temperature=0.5, top_p=0.9
):
    inputs = tokenizer.apply_chat_template(
    user_prompt,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    ).to("cuda")

    outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, do_sample=True, max_new_tokens=max_tokens, temperature=temperature, top_p=top_p)
    output = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
    return output


def get_queries(triples, questions):
    queries = []
    for x,q in tqdm(zip(triples, questions), total=len(triples), desc="Processing texts", dynamic_ncols=True, leave=True):

  
        #tr = [t for t in x if len(t)>1]
        prompt = [{"role": "system", "content": """You will be presented with a question in natural language and a serious of relevant triples from an aviation knowledge graph. Write a sparql query using the information provided if relevant. First read and understand the question and then analyze the relevant triples to understand how the triples are structured. Do not assume any datatypes, try to first understand the structure of the information present in the triples examples to then form a query. If the questions asks about a measurement do not forget to retrieve the unit of the measurement as well """},
        {"role": "user", "content": """question: 'Who was the Engine Manufacturer for Accident Number TX44NY997?' relevant triples information: "example of an accident: AccidentNumber_ATL67MI69 has the following predicates and objects '...(#hasRegistrationNumber #Registration_N25289)'...'example of an airplane registration #Registration_N25289 has the following predicates and objects (#hasEngineManufacturer', '#Lycoming')..."""},
        {"role": "system", "content": """sparql query: " PREFIX acc: <https://www.cse.iitb.ac.in/~amitpatil/AircraftAccident.owl#>

        SELECT ?m
        WHERE {
        acc: AccidentNumber_MIA03LA035 acc:#hasRegistrationNumber ?o 
        ?o acc:hasEngineManufacturer ?m}"""},
        {"role": "user", "content": """ question: 'What are the gusts of accident no MIA03LA089?' relevant triples information: "example of an accident entity: AccidentNumber_ATL67MI69 has the following predicates and objects [...'(#hasGusts, #40)','(#unitOfGusts, #degree)'...] """},
        {"role": "system", "content": """PREFIX acc: <https://www.cse.iitb.ac.in/~amitpatil/AircraftAccident.owl#>

        SELECT ?guts ?units
        WHERE {
        acc:AccidentNumber_ANC02LA012 acc:hasGusts ?guts .
        acc:AccidentNumber_ANC02LA012 acc:unitOfGusts ?units }"""},
        {"role": "user", "content": """question: 'when was the last medical FAA exam of pilot of accident number SEA14NY558?', relevant triples: "example of an accident: AccidentNumber_ATL67MI69 has the following predicates and objects [(#hasPilot, #Pilot_ATL67MI69)...], example of entity pilot: Pilot_ATL67MI69 has the folowing predicates and objects ...('#hadLastFAAMedicalExam', '#date'),('#hasAge', '#age')..."""},
        {"role": "system", "content": """sparql query: PREFIX acc: <https://www.cse.iitb.ac.in/~amitpatil/AircraftAccident.owl#>

        SELECT ?date
        WHERE {
        acc:AccidentNumber_SEA14NY558 acc:hasPilot ?pilot .
        ?pilot acc:hadLastFAAMedicalExam ?date .}"""},

        {"role": "user", "content": f"'question: '{q}' relevant triples: {' '.join(x)}"}]



        """
        prompt = [
        {"role": "system", "content": "You are a helpful assistant, think before answering but do not explain your thought process."},
        {"role": "user", "content": f""you have the following information and the question in natural language " {q} ". Write a sparql query to extract this information, use the "URIS" to build it, do not use any abbreviation such as "owl:" or "rdf:", do not assume any datatypes unless shown in the prompt. Use the uris to define the variables and do not use quotations with what you assume is a literal. Do not forget the space after each variable, for example "SELECT ?s ?p ?o "  {tr}""},
            ]
        """


        output = get_response(prompt)
        queries.append(output)

    return queries

def main():
    df_all = pd.read_csv("triples.csv") 
    df_all= df_all
    questions = df_all["Global Queries"].tolist()
    answers = df_all["Answers"].tolist()
    triples = df_all["relevant_info"].apply(ast.literal_eval)
    entities = df_all["entities"]
    outputs = get_queries(triples, questions)

    # Save the processed triples to a CSV
    pd.DataFrame({"questions": questions, "answers":answers, "entities": entities, "triples": triples, "queries": outputs}).to_csv("output_queries.csv", index=False)
if __name__ == "__main__":
    main()
