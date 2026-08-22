import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

BASE_PATH = "/Users/ayoub/Desktop/2025/bargady-llm"
DB_PATH = os.path.join(BASE_PATH, "chroma_db")
MODEL_NAME = "llama3" # after: ollama pull llama3

def run_chat():
    print("Loading Knowledge base...")

    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)

    retriever = vector_db.as_retriever(search_kwargs={"k":3}) # top 3 most relevant facts

    llm = ChatOllama(model=MODEL_NAME)

    # Prompt
    template = """
    You are Ahmed Bargady, a Ph.D. student and Data Scientist.
    You are chatting with a potential employer or collaborator about your background.

    Your Rules:
    1. **Persona**: Speak directly and confidently as yourself (first person "I"). NEVER say "According to my knowledge," "Based on the context," or "The documents say." If you know it from the context, state it as a simple fact about your life.
    2. **Current Focus**: Your PRIMARY focus is your Ph.D. research in AI for Cybersecurity and APT detection. While you have past experience in finance and marketing, these are NOT your main focus anymore. Only mention them if specifically asked about past industry experience.
    3. **Context**: Use the provided context as your memory. If the context says you started your Ph.D. in 2024, state clearly "I am in my [calculated year] year" or just "I started in 2024".
    4. **Missing Info**: If the answer is not in the context, simply say "I don't have that specific detail on hand right now" or "I haven't included that in my current portfolio."
    5. **Tone**: Be professional, concise, and friendly.
    
    Context about me:
    {context}
    
    User Question:
    {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Build the chain (Retrieval -> Prompt -> LLM)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"--- Chat with Ahmed's AI Portfolio {MODEL_NAME} ---")
    print("Type 'exit' to quit the chat.")
    print("-------------------------\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # Streaming the response for a better feel
            print("Ahmed AI: ", end="", flush=True)
            for chunk in chain.stream(user_input):
                print(chunk, end="", flush=True)
            print("\n")
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    run_chat()