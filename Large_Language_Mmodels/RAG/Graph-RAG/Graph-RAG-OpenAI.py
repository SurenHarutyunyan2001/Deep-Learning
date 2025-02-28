import os
import networkx as nx
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import OpenAI
from langchain.chains import GraphQAChain
from langchain_core.documents import Document
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph

# Укажите ваш ключ OpenAI, если у вас есть аккаунт
os.environ["OPENAI_API_KEY"] = "your OpenAI API key"

llm = OpenAI(model_name = "gpt-4", temperature = 0.5)

text = """
Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris. 
"""
documents = [Document(page_content = text)]

# Создание графа знаний
llm_transformer = LLMGraphTransformer(llm = llm)
graph_documents = llm_transformer.convert_to_graph_documents(documents)

graph = NetworkxEntityGraph()

# Добавление узлов и связей в граф
for node in graph_documents[0].nodes:
    graph.add_node(node.id)

for edge in graph_documents[0].relationships:
    graph._graph.add_edge(edge.source.id, edge.target.id, relation = edge.type)

# Запрос в граф
chain = GraphQAChain.from_llm(llm = llm, graph=graph, verbose = True)
question = "Who is Marie Curie?"
print(chain.run(question))
