import os
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain import PromptTemplate
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

EVAL_PROMPT = """
Réponse Attendue: {expected_response}
Réponse Réelle: {actual_response}
---
(Répondez par 'vrai' ou 'faux') La réponse réelle correspond-elle à la réponse attendue ?
"""

def load_and_prepare_file(file_path):
    loader = TextLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(all_splits, local_embeddings)
    return vectorstore

def query_logs(vectorstore, query):
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

    llm = Ollama(model="llama3.1")
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    result = qa_chain({"query": query})
    return result.get('result', 'No result found.')

def query_and_validate(vectorstore, question: str, expected_response: str):
    response_text = query_logs(vectorstore, question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="llama3.1")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "vrai" in evaluation_results_str_cleaned:
        print("\033[92m" + f"Réponse: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "faux" in evaluation_results_str_cleaned:
        print("\033[91m" + f"Réponse: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Résultat d'évaluation invalide. Impossible de déterminer si 'vrai' ou 'faux'."
        )

def run_tests(log_file_path):
    vectorstore = load_and_prepare_file(log_file_path)

    tests = [
        ("Quelle est l'adresse IP la plus fréquemment mentionnée dans les logs ? (Répondez uniquement avec l'adresse IP)", "10.0.0.150"),
        ("Combien d'erreurs 404 sont enregistrées dans les logs ? (Répondez uniquement avec le nombre)", "23"),
        ("Combien de tentatives de connexion échouées y a-t-il eu ? (Répondez uniquement avec le nombre)", "12"),
        ("Quel utilisateur a effectué le plus d'actions selon les logs ? (Répondez uniquement avec le nom d'utilisateur)", "admin"),
        ("À quelle heure le trafic a-t-il atteint son pic ? (Répondez au format HH:00)", "15:00"),
        ("Y a-t-il eu des tentatives d'injection SQL détectées ? (Répondez par Oui ou Non)", "Oui"),
    ]

    for question, expected_response in tests:
        print(f"\nTest: {question}")
        result = query_and_validate(vectorstore, question, expected_response)
        if result:
            print("\033[92m" + "Test réussi!" + "\033[0m")
        else:
            print("\033[91m" + "Test échoué." + "\033[0m")

if __name__ == "__main__":
    log_directory = "./logs"
    log_files = [f for f in os.listdir(log_directory) if f.endswith('.md')]
    
    if not log_files:
        print("Aucun fichier de log trouvé dans le répertoire ./logs")
    else:
        log_file_path = os.path.join(log_directory, log_files[0])
        print(f"Utilisation du fichier de log : {log_file_path}")
        run_tests(log_file_path)