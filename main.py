from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain import PromptTemplate
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
import sys
import os
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text

# Initialize Rich console
console = Console()

class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# Load the logs and split them into chunks
loader = TextLoader("./logs1.md")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

with SuppressStdout():
    local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(all_splits, local_embeddings)

# Create a retrieval-based QA chain
while True:
    query = Prompt.ask("[bold green]Enter your query (or type 'exit' to quit):[/bold green]")
    if query.lower() == "exit":
        console.print("[bold red]Exiting...[/bold red]")
        break
    if query.strip() == "":
        console.print("[bold yellow]Empty query, please try again.[/bold yellow]")
        continue

    # Prompt Template
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # Initialize the LLM with callback for streaming output
    llm = Ollama(model="llama3.1", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    console.print("[cyan]Retrieving relevant logs and generating response...[/cyan]")
    
    result = qa_chain({"query": query})

    # Extract the LLM response from the result (assuming it's in a 'result' field)
    response_text = result['result'] if isinstance(result, dict) else str(result)

    # Display the result in a rich panel
    console.print(Panel(Text(response_text, style="bold white"), title="[bold cyan]Log Analysis Result[/bold cyan]"))
