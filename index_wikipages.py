# REPLACE THIS WITH YOUR CODE
import pydantic
import openai
from pydantic import BaseModel, Field
from typing import List
import wikipedia 

# Core LlamaIndex modules
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.program.openai import OpenAIPydanticProgram

wikipedia.set_user_agent("WikiIndexerBot/1.0 (contact: prabhudeengtm20@gmail.com)")
# Custom utility function from Task 1
from utils import get_apikey

# define the data model in pydantic
class WikiPageList(BaseModel):
    "Data model for WikiPageList"
    pages: List[str] = Field(description="A list of specific Wikipedia page titles extracted from the query.")


def wikipage_list(query):
    openai.api_key = get_apikey()

    # Create a prompt template to guide the model toward extracting the requested titles
    prompt_template_str = (
        "You are an assistant that extracts entity names from text requests to index Wikipedia pages.\n"
        "Extract all the requested Wikipedia topics, cities, or pages from the following query "
        "and format them as a clean list of titles.\n"
        "Query: {query}\n"
    )

    # Initialize an OpenAIPydanticProgram object
    program = OpenAIPydanticProgram.from_defaults(
        output_cls=WikiPageList,
        prompt_template_str=prompt_template_str,
        verbose=False
    )

    wikipage_requests = program(query=query)

    return wikipage_requests.pages


def create_wikidocs(wikipage_requests):
    reader = WikipediaReader()
    documents = []
    for page in wikipage_requests:
        try:
            documents.extend(reader.load_data(pages=[page]))
        except wikipedia.exceptions.DisambiguationError as e:
            if not e.options:
                print(f"Skipping '{page}': disambiguation error with no options")
                continue
            resolved = e.options[0]
            print(f"'{page}' is ambiguous, using first option: '{resolved}'")
            try:
                documents.extend(reader.load_data(pages=[resolved]))
            except Exception as inner_e:
                print(f"Skipping '{resolved}': {inner_e}")
        except wikipedia.exceptions.PageError:
            print(f"Skipping '{page}': page not found")

    return documents


def create_index(query):
    global index
    # 1. Get the list of requested pages from the query string
    pages_to_load = wikipage_list(query)
    
    # 2. Ingest the actual raw documents from Wikipedia
    documents = create_wikidocs(pages_to_load)
    
    # 3. Create a SentenceSplitter object defining the chunk_size and chunk_overlap
    splitter = SentenceSplitter(chunk_size=150, chunk_overlap=45)
    
    # 4. Use the parser to split documents down into concrete text nodes
    nodes = splitter.get_nodes_from_documents(documents)
    
    # 5. Build the VectorStoreIndex over the parsed nodes
    print(f"Total Vector Store Nodes: {len(nodes)}")
    index = VectorStoreIndex(nodes)

    return index




if __name__ == "__main__":
    query = "/get wikipages: paris, america people, indian people"
    index = create_index(query)
    print("INDEX CREATED", index)
