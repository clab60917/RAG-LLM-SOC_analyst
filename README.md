# 🔍 SOC Analyst Level 1 Replacement using a RAG LLM 🚀

Welcome to the **SOC Analyst Level 1 Replacement using RAG LLM** project! This repository presents a small research-oriented Proof of Concept (POC) aimed at exploring the feasibility of using a Retrieval-Augmented Generation (RAG) Large Language Model (LLM) to replace or assist a Level 1 SOC (Security Operations Center) Analyst.

## 📜 Project Overview

Security Operations Centers are the backbone of cybersecurity in organizations, continuously monitoring and analyzing data to detect potential threats. However, the increasing volume of security logs and alerts can overwhelm human analysts, particularly those at Level 1, who are responsible for initial triage and response.

This project explores the potential of using an LLM, combined with a retrieval system, to automate some of the tasks typically performed by a Level 1 SOC analyst. By leveraging advanced natural language processing (NLP) techniques, the system can answer queries related to server logs and provide actionable insights.

## 🛠️ Technology Stack

- **LangChain**: Utilized for orchestrating the retrieval-augmented generation (RAG) pipeline.
- **Ollama LLM**: The LLM backbone, capable of understanding and processing natural language queries.
- **Chroma**: A vector store for efficient retrieval of relevant log information.
- **Python**: The core language used for implementation.
- **Pandas & Matplotlib** (Optional): For potential future extensions involving data analysis and visualization.

## ⚙️ How It Works

1. **Log Ingestion**: The system loads and processes server logs stored in a Markdown file (`logs1.md`). The logs are split into manageable chunks for efficient processing.

2. **Vectorization**: Each chunk of log data is embedded into a vector space using the `OllamaEmbeddings` model. This allows for efficient similarity searches.

3. **Query Processing**: Users can input natural language queries, such as "What are the suspicious activities in the logs?" The system retrieves relevant log information and uses the LLM to generate a concise and contextually accurate response.

4. **Response Generation**: The system provides a response based on the retrieved context, simulating the role of a Level 1 SOC analyst by answering queries about the logs.

## 📂 Project Structure

```
├── logs1.md                     # Sample log data
├── main.py                      # Main Python script implementing the POC
├── README.md                    # Project documentation (you are here!)
└── requirements.txt             # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

Before you start, ensure you have the following installed:

- Python 3.8+
- Virtual environment tools (optional but recommended)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/clab60917/RAG-LLM-SOC_analyst.git
   cd RAG-LLM-SOC_analyst
   ```

2. **Create a virtual environment (optional):**

   ```bash
   python -m venv env
   source env/bin/activate  
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the POC:**

   ```bash
   python main.py
   ```

### Usage

- **Querying the Logs**: Once the script is running, you can start querying the logs. Type your query and press enter. For example:
  - `Query: What are the most recent suspicious activities?`
  - `Query: Summarize the failed login attempts.`

- **Exit**: To exit the script, simply type `exit`.

## 📈 Future Work

This POC lays the groundwork for a more comprehensive system capable of fully automating Level 1 SOC operations. Future enhancements might include:

- **Real-time Log Streaming**: Integrate with live data sources for real-time analysis.
- **Advanced Analytics**: Implement graph-based and statistical analysis of log data.
- **Actionable Responses**: Automate responses such as blocking IP addresses or triggering alerts.

## 🧠 Research Implications

This project is part of an ongoing small research initiative. The ultimate goal is to evaluate whether RAG-based LLMs can efficiently scale the capabilities of SOC teams, reducing the workload on human analysts and enabling faster, more accurate incident response.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to fork the repository and submit a pull request.

## 🙏 Acknowledgements

Special thanks to the creators of [LangChain](https://github.com/hwchase17/langchain), [Ollama](https://www.ollama.ai/), and the open-source community for providing the tools and frameworks that made this project possible.

---

**👤 Author:** [Clab60917](https://github.com/clab60917)  

---