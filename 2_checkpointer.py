from dotenv import load_dotenv
import uuid

from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain.messages import HumanMessage, AIMessage

from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

load_dotenv()

llm = init_chat_model('openai:gpt-4o-mini')

def prompt_llm(state: MessagesState):
    response = llm.invoke(state['messages'])

    return {'messages': [response]}

graph_builder = StateGraph(MessagesState)

graph_builder.add_node(prompt_llm)
graph_builder.add_edge(START, 'prompt_llm')
graph_builder.add_edge('prompt_llm', END)

graph = graph_builder.compile(checkpointer= checkpointer)

config = {
    'configurable': {
        'thread_id' : uuid.uuid4()
    }
}


while True:
    user_message = input('Enter Message: ')
    print(graph.invoke(
        {'messages':[HumanMessage(content=user_message)]},
        config= config
    ))