import os

from dotenv import load_dotenv

from langchain.chains import ConversationalRetrievalChain
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embedding_function = OpenAIEmbeddings()
llm = ChatAnthropic(model_name="claude-3-opus-20240229", api_key=os.getenv("CLAUDE_API"))
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# Create conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
retriever = db.as_retriever()

# Create conversation chain with knowledge graph
conversation = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=False,
)

print("Welcome to the Smart Bank Chatbot! Type 'exit' to end the conversation.")

chat_history = []

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    # Send user input to the conversation chain
    response = conversation.invoke({"question": user_input, "chat_history": chat_history})
    print(f"SmartBank: {response['answer']}")
    chat_history = [(user_input, response["answer"])]
