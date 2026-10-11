from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain.messages import HumanMessage, AIMessage

load_dotenv()

llm = init_chat_model('openai:gpt-4o-mini')

def prompt_llm(state: MessagesState):
    response = llm.invoke(state['messages'])

    return {'messages': [response]}

graph_builder = StateGraph(MessagesState)

graph_builder.add_node(prompt_llm)
graph_builder.add_edge(START, 'prompt_llm')
graph_builder.add_edge('prompt_llm', END)

graph = graph_builder.compile()
ai_msg = 'Hello! How can I assist you today?\n'
user_msg = input(ai_msg)
while True:
    messages = [HumanMessage(user_msg)]
    response = graph.invoke({'messages': messages })
    messages.append(AIMessage(response['messages'][1].content))
    print(response['messages'][1].content)
    human_message = input('Enter Text: ')
    messages.append(HumanMessage(human_message))