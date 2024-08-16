from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain import PromptTemplate
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter  # <-- Import added here
import os
import sys
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

def load_and_prepare_file(file_path):
    # Load the selected log file and split it into chunks
    loader = TextLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    with SuppressStdout():
        local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = FAISS.from_documents(all_splits, local_embeddings)

    return vectorstore

def select_file(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not files:
        console.print("[bold red]No log files found in the directory.[/bold red]")
        return None

    console.print("[bold cyan]Available log files:[/bold cyan]")
    for i, file in enumerate(files, start=1):
        console.print(f"[bold green]{i}[/bold green]: {file}")

    file_index = Prompt.ask("[bold green]Enter the number of the file you want to use:[/bold green]", choices=[str(i) for i in range(1, len(files) + 1)])
    
    selected_file = files[int(file_index) - 1]
    return os.path.join(directory, selected_file)

# Directory containing log files
log_directory = "./logs"

# Create a retrieval-based QA chain
while True:
    # Select the log file
    selected_file_path = select_file(log_directory)
    if not selected_file_path:
        continue

    # Prepare the selected file
    vectorstore = load_and_prepare_file(selected_file_path)

    # Ask for the query
    query = Prompt.ask("[bold green]Enter your query (or type 'exit' to quit):[/bold green]")
    if query.lower() == "exit":
        console.print("[bold red]Exiting...[/bold red]")
        break
    if query.strip() == "":
        console.print("[bold yellow]Empty query, please try again.[/bold yellow]")
        continue

    # Prompt Template
    template = """You are a level 1 SOC analyst. Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer precise and as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # Initialize the LLM without the streaming output handler
    llm = Ollama(model="llama3.1")
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    console.print("[cyan]Retrieving relevant logs and generating response...[/cyan]")
    
    result = qa_chain({"query": query})

    # Assuming result is a dict and contains the response in the 'result' key
    response_text = result.get('result', 'No result found.')

    # Display the result in a rich panel
    console.print(Panel(Text(response_text, style="bold white"), title="[bold cyan]Log Analysis Result[/bold cyan]"))

    # Add a line break between questions
    console.print()
