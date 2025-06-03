import pandas as pd
import ast
from rdflib import Graph
import json
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool
from tqdm import tqdm
from urllib.parse import urlparse

#load premade triples

with open('knowledge_graphs/premade_triples.json') as json_file:
    premade_queries = json.load(json_file)


# Load RDF Graph
g = Graph()
g.parse("knowledge_graphs/entities/Aviation_KG.owl", format="xml")

#function to create the query

def extract_name(uri):
    """Extracts the last part of a URI after # or /"""
    parsed = urlparse(uri)
    return uri.split("#")[-1] if "#" in uri else parsed.path.split("/")[-1]

def posprocess_output(output):

    return [( f"#{extract_name(p)}", f"#{extract_name(o)}") for p, o in output]

def fetch_query(pred, obj):
    query = f"""
    SELECT DISTINCT ?s ?predicate ?o
    WHERE {{
        {{
            SELECT (SAMPLE(?s) AS ?s) ?predicate (SAMPLE(?o) AS ?o)
            WHERE {{
                ?s ?predicate ?o.
                FILTER({pred})
            }}
            GROUP BY ?predicate
        }}
        UNION
        {{
            SELECT (SAMPLE(?s) AS ?s) (SAMPLE(?predicate) AS ?predicate) ?o
            WHERE {{
                ?s ?predicate ?o.
                FILTER({obj})
            }}
            GROUP BY ?o
        }}
    }}
    """
    #print(query)
    return query



#Function to get triples


def get_object(entity):
    """Fetch RDF triples where entity appears as an object, checking word pairs first."""
    words = entity.split()
    num_words = len(words)
    #print(words, num_words)
    if num_words > 1:
        
        # Step 1: Try full entity match with variations
        entity_variations = [entity.replace(" ", ""), entity.replace(" ", "_")]
        filter_clause_p = " || ".join([f'CONTAINS(LCASE(STR(?predicate)), "{var.lower()}")' for var in entity_variations])
        filter_clause_o = " || ".join([f'CONTAINS(LCASE(STR(?o)), "{var.lower()}")' for var in entity_variations])
    else:
        # If the entity is only one word just check for that word
        filter_clause_p = f'CONTAINS(LCASE(STR(?predicate)), "{entity.lower()}")'
        filter_clause_o = f'CONTAINS(LCASE(STR(?o)), "{entity.lower()}")'
    

   
    query = fetch_query(filter_clause_p, filter_clause_o)
    
    try:
        result = g.query(query)
        
        if len(list(result)) >0:
            #print("yes")
            return [x for x in result]
    except Exception as e:
        print(f"SPARQL Query Error in first attempt: {e}")
    if num_words >1:
        # Step 2: Try matching word pairs (sliding window)
        for i in range(num_words - 1):  # Generate pairs
            pair = f"{words[i]} {words[i+1]}"
            #print(pair)
            pair_filter_clause_p = " && ".join([f'CONTAINS(LCASE(STR(?predicate)), "{p.lower()}")' for p in pair.split()])
            pair_filter_clause_o = " && ".join([f'CONTAINS(LCASE(STR(?o)), "{p.lower()}")' for p in pair.split()])
            #print(pair_filter_clause_o)

            query = fetch_query(pair_filter_clause_p, pair_filter_clause_o)
            try:
                result = g.query(query)
                if len(list(result)) >0:
                    d = {entity: {f"The entity {entity} has the following relevant triples: ":[(f"#{extract_name(s)}", f"#{extract_name(p)}", f"#{extract_name(o)}") for s, p, o in result]}}
                    return d
            except Exception as e:
                print(f"SPARQL Query Error for pair {pair}: {e}")
                print(query)
        
        for word in words:
            if word in premade_queries:
                return premade_queries[word]  # Return the dictionary value



        # Step 3: If no pairs match, try individual words
        word_filters_p = " || ".join([f'CONTAINS(LCASE(STR(?o)), "{word.lower()}")' for word in words])
        word_filters_o = " || ".join([f'CONTAINS(LCASE(STR(?o)), "{word.lower()}")' for word in words])

        query = fetch_query(word_filters_p, word_filters_o)

        try:
            result = g.query(query)
            if len(list(result)):
                return [x for x in result]
        except Exception as e:
            print(f"SPARQL Query Error in last attempt: {e}")

    return []

def process_entity(entity):

    if entity.lower() in premade_queries:
        return premade_queries[entity.lower()]
    else:
        return get_object(entity)

def process_entity_list(entity_list):
    """Process a list of entities and return a sublist of triples for each entity."""
    #print(entity_list)
    return [process_entity(entity) for entity in entity_list]

def get_triples(list_entities):
    output = []
    for x in tqdm(list_entities, leave=True):
        output.append(process_entity_list(x))
    return output

#nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
def lemmatize(text):
  output = []
  for t in text:
    doc = t.split()
    lemmatized_text = " ".join([lemmatizer.lemmatize(token)for token in doc])
    output.append(lemmatized_text)
  return output


def main():
    
   
    df_entities = pd.read_csv("knowledge_graphs/entities/entities_llama3-70b_fixed.csv")

    answers = df_entities["Answers"].tolist()
    entities = df_entities["only_entities"].apply(ast.literal_eval)  
    entities_lemmatized = [lemmatize(x) for x in entities]
    triples = get_triples(entities_lemmatized)
    questions = df_entities["Global Queries"].tolist()


    # Save the processed triples to a CSV
    pd.DataFrame({"questions": questions, "answers":answers, "entities": entities, "triples": triples}).to_csv("output_triples_full_2503.csv", index=False)

if __name__ == "__main__":
    main()