import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
import os

def LangGraph_run():

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    graph_builder = StateGraph(State)

    model_name = "gpt-4o-mini-2024-07-18"

    llm = ChatOpenAI(
        model = model_name,
        api_key = st.session_state.api_key
    )


    # 여기에 response를 만들때 RAG를 넣으면 될듯.
    def chatbot(state: State):
        
        syllabus_prompt = """
        You are an assistant specialized in providing answers related to the syllabus of Solid Mechanics course on Spring 2025.

        Using the provided syllabus file, your task is to answer questions about the coursework.
        ==================================

        When answering questions, please refer to the syllbus' content to provide accurate and step-by-step guidance. 
        Ensure that you clearly explain the specific instructions for each topic based on the student’s query.

        Document: {context}

        Answer format:

        Answer:
        Provide the detailed answer based on the syllabus, focusing on the section relevant to the student's query.
        """
        
        vectorstore_path = 'syllabus_vectorstore'
        vectorstore = FAISS.load_local(vectorstore_path, embeddings=OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        retriever = retriever = vectorstore.as_retriever()

        prompt = ChatPromptTemplate.from_messages([
            ("system", syllabus_prompt),
            ("user", "{question}")
            ])

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
        )

        result = chain.invoke(str(state["messages"]))
        return {"messages": [result]}
    
    
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile()

    # try:
    #     graph_bytes = graph.get_graph().draw_mermaid_png()
    #     st.image(graph_bytes, caption = "Chatbot Graph")
    # except Exception as e:
    #     st.error(f"Failed to dislay graph: {e}")

    if "messages_01" not in st.session_state:
        st.session_state.messages_01 = []

    # Display all previous messages
    for message in st.session_state.messages_01:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get new user input and process it
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages_01.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare the entire chat history to send to the model
        full_conversation = [(msg["role"], msg["content"]) for msg in st.session_state.messages_01]

        # Generate response using the model with the full conversation history
        for event in graph.stream({"messages": full_conversation}):
            for value in event.values():
                # Access the content directly from the AIMessage object
                response = value["messages"][-1].content
                st.session_state.messages_01.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)



def main():
    # Sidebar to reset session state
    with st.sidebar:
        if st.button("Reset Session"):
            st.session_state.clear()
            # st.experimental_rerun()  # Rerun the app to reflect the changes

    # Page title
    st.title("Syllabus Chatbot")

    if "api_key_submitted" not in st.session_state:
        st.session_state.api_key_submitted = False

    if not st.session_state.api_key_submitted:
        api_key = st.text_input("Please input your OpenAI API Key:", type="password")

        if st.button("Submit"):
            if api_key:
                st.session_state.api_key = api_key
                os.environ["OPENAI_API_KEY"] = api_key

                st.session_state.api_key_submitted = True

                st.success("✅ API Key submitted successfully! You can now use the chatbot.")
            else:
                st.warning("Please input your API key")

    if st.session_state.api_key_submitted:
        LangGraph_run()

    # api_key = st.text_input("API 키를 입력하세요:", type="password")

    # if api_key:
    #     st.session_state["api_key"] = api_key

    # After API key submission, start the chatbot interaction
    # if "api_key" in st.session_state:
    #     LangGraph_run()

# 다른 파일에서 이 코드를 import 해도 자동 실행 되는 것을 방지.
if __name__ == "__main__":
    main()



