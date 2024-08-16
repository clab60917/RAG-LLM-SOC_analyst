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
    Keep the answer precise and as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    llm = Ollama(model="phi3:medium-128k")
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

    if "vrai" in evaluation_results_str_cleaned:
        return True, response_text
    elif "faux" in evaluation_results_str_cleaned:
        return False, response_text
    else:
        raise ValueError(
            f"Résultat d'évaluation invalide. Impossible de déterminer si 'vrai' ou 'faux'."
        )

def run_tests(log_file_path):
    vectorstore = load_and_prepare_file(log_file_path)

    tests = [
        ("Combien de tentatives de connexion ont échoué ?", "9"),
        ("Combien d'attaques de force brute ont été détectées ?", "8"),
        ("Combien de tentatives d'injection SQL ont été détectées ?", "5"),
        ("Combien d'attaques DDoS ont été détectées ?", "5"),
        ("Combien de tentatives de téléversement de fichiers suspects ont été détectées ?", "6"),
        ("Quelle IP a tenté d'exécuter du code à distance ?", "192.168.1.100"),
        ("Quelle IP a effectué la plupart des tentatives d'attaque ?", "203.0.113.1"),
    ]

    passed_tests = 0
    failed_tests = 0

    for question, expected_response in tests:
        print(f"\nTest: {question}")
        try:
            result, actual_response = query_and_validate(vectorstore, question, expected_response)
            if result:
                passed_tests += 1
                print("\033[92m" + "Test réussi!" + "\033[0m")
            else:
                failed_tests += 1
                print("\033[91m" + f"Test échoué. Réponse attendue: {expected_response}, Réponse réelle: {actual_response}" + "\033[0m")
        except Exception as e:
            failed_tests += 1
            print("\033[91m" + f"Erreur lors du test: {str(e)}" + "\033[0m")

    print("\n" + "="*40)
    print(f"\033[94mTotal des tests réussis: {passed_tests}\033[0m")
    print(f"\033[94mTotal des tests échoués: {failed_tests}\033[0m")
    print("="*40)

if __name__ == "__main__":
    log_directory = "./logs"
    log_files = [f for f in os.listdir(log_directory) if f.endswith('.md')]
    
    if not log_files:
        print("Aucun fichier de log trouvé dans le répertoire ./logs")
    else:
        log_file_path = os.path.join(log_directory, log_files[0])
        print(f"Utilisation du fichier de log : {log_file_path}")
        run_tests(log_file_path)
