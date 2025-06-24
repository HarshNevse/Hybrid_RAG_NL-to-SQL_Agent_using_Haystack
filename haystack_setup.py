import re
def clean_text(text):
    #text = re.sub(r"\n+", "\n", text)  #excessive newlines
    #text = re.sub(r"_+", " ", text)
    #text = re.sub(r"\s+", " ", text)  # excessive spaces
    #text = re.sub(r"^\s+|\s+?$", "", text)  # leading/trailing spaces
    return text


# In[221]:


from haystack import component, Document
from typing import List
print("Initializing splitter...", flush=True)
from haystack.components.preprocessors import DocumentSplitter
splitter = DocumentSplitter(split_by='line', split_length = 10, split_overlap = 0)
# In[222]:

print("Initializing doc-embedder...", flush = True)
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

# Document Embedder
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()  # Optional, helps speed up the first batch processing (for anyone reading this, comment it out after first load)


# In[223]:

print("Initializing doc_store and writer...", flush = True)
from haystack.document_stores.in_memory import InMemoryDocumentStore
document_store = InMemoryDocumentStore()

from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
document_writer = DocumentWriter(document_store, policy=DuplicatePolicy.SKIP)


# In[224]:


from haystack import Pipeline

em_pipeline = Pipeline()
em_pipeline.add_component(instance=splitter, name="document_splitter")
em_pipeline.add_component(instance=doc_embedder, name="document_embedder")
em_pipeline.add_component(instance=document_writer, name="document_writer")

em_pipeline.connect("document_splitter.documents", "document_embedder.documents")
em_pipeline.connect("document_embedder.documents", "document_writer.documents")

print("em_pipeline ready to run.", flush = True)
# In[225]:


#em_pipeline.show()


# In[226]:


#r = em_pipeline.run({'document_splitter' : {'sql_schema' : text}})


# In[227]:


len(document_store.storage)


# In[238]:

print('Initializing text embedder...',flush=True)
from haystack.components.embedders import SentenceTransformersTextEmbedder

text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
text_embedder.warm_up()


# In[248]:

print('Initializing retrievers...', flush = True)
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever

bm25_retriever = InMemoryBM25Retriever(document_store, top_k = 7)
embedding_retriever = InMemoryEmbeddingRetriever(document_store, top_k = 7)


# In[249]:


@component
class CustomPromptBuilder:
    """
    A custom component that formats the documents and question into a prompt string.
    """
    @component.output_types(prompt=str)
    def run(self, documents: List[Document], question: str):
        # joining document contents into a single string
        context = "\n".join([doc.content for doc in documents])
  
        prompt = f"""
    You are a professional SQL developer. Your task is to generate only accurate SQL queries based strictly on the given database schema and question. **Your response must consist of a valid SQL query only, with no extra explanations or comments.**

Strict Rules:
1. Use only the provided database schema or context. **Do not invent or assume schema details.**
2. If the context contains enough information, generate the most precise and efficient SQL query possible.
3. Ensure correct use of table joins when required—**never ignore relational context**.
4. Carefully verify all table and column names against the schema before generating the query.
5. If the answer cannot be derived from the schema, respond only with: `I don't know`
6. Do not provide any output besides the SQL query or `I don't know`. **No comments, no explanations.**

Context:
- The provided documents contain the database schema and relevant metadata.
- Use this schema to understand table structures, column types, and relationships.

Expected Output:
- One of:
  - A complete, syntactically correct SQL query that answers the question.
  - The exact phrase: `I don't know` if the query cannot be answered from the schema.

Requirements:
- Output **only the SQL query** (or `I don't know`) as plain text. No markdown, no extra formatting.
- Ensure professional SQL standards: proper use of joins, aliases, conditions, and clear formatting.
- do not enclose the sql query in sql docstring

Inputs:
---
**Relevant Context schema:**
{clean_text(context)}

**Question:** {question}
**Answer:**
    """
        return {"prompt": prompt}


custom_prompt_builder = CustomPromptBuilder()


# In[241]:

print('Initializing LLM...',flush=True)
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

chat_generator = OpenAIGenerator(
    api_key=Secret.from_env_var("GROQ_API_KEY"),
    api_base_url="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
    generation_kwargs = {"max_tokens": 512}
)


# In[242]:


from haystack.components.joiners import DocumentJoiner

joiner = DocumentJoiner()


# In[243]:
print('Initializing ranker...',flush=True)
from haystack.components.rankers import TransformersSimilarityRanker

ranker = TransformersSimilarityRanker(top_k=11)
ranker.warm_up()


# In[244]:


rag_pipeline = Pipeline()

rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("bm25_retriever", bm25_retriever)
rag_pipeline.add_component("embedding_retriever", embedding_retriever)
rag_pipeline.add_component("joiner", joiner)
rag_pipeline.add_component("ranker", ranker)
rag_pipeline.add_component("custom_prompt_builder", custom_prompt_builder)
rag_pipeline.add_component("llm", chat_generator)


# In[245]:


rag_pipeline.connect("bm25_retriever", "joiner")
rag_pipeline.connect("text_embedder", "embedding_retriever")
rag_pipeline.connect("embedding_retriever", "joiner")
rag_pipeline.connect("joiner", "ranker")
rag_pipeline.connect("ranker", "custom_prompt_builder")
rag_pipeline.connect("custom_prompt_builder","llm")

print('rag_pipeline ready to run.', flush=True)

def ask(question: str):
    r = rag_pipeline.run(
    {
    "text_embedder": {"text": question},
    "custom_prompt_builder": {"question": question},
    "bm25_retriever": {"query": question},
    "ranker": {"query": question}
    },
    include_outputs_from="custom_prompt_builder"
                    )

    finished_prompt = r['custom_prompt_builder']['prompt']
    llm_SQL = r["llm"]["replies"][0]

    return {'final_prompt' : finished_prompt, 'sql' : llm_SQL}
