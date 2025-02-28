import networkx as nx
import matplotlib.pyplot as plt

# Создаем граф
G = nx.DiGraph()

# Добавлям узлы и ребра
G.add_node("RAG", type = "Architecture")
G.add_edge("RAG", "search and text generation", relation = "combines")
G.add_edge("RAG", "knowledge base", relation = "uses")
G.add_edge("knowledge base", "relevant information", relation = "to find")
G.add_edge("relevant information", "response", relation = "before generating")
G.add_edge("RAG", "Google", relation="introduced by")
G.add_edge("Google", "2020", relation = "in")
G.add_edge("RAG", "chatbots", relation = "used in")
G.add_edge("RAG", "search engines", relation = "used in")
G.add_edge("RAG", "AI assistants", relation = "used in")
G.add_edge("RAG", "retrieval-based components", relation = "combines")
G.add_edge("RAG", "generation-based components", relation = "combines")
G.add_edge("retrieval-based components", "better answers", relation = "to provide")
G.add_edge("generation-based components", "better answers", relation = "to provide")
G.add_edge("RAG", "Search Component", relation = "has component")
G.add_edge("Search Component", "relevant context", relation = "finds")
G.add_edge("relevant context", "document base", relation = "from")
G.add_edge("RAG", "Generation Component", relation = "has component")
G.add_edge("Generation Component", "answer", relation = "creates")
G.add_edge("answer", "context", relation = "from")

G.add_node("Irrelevant Sentences", type="Evaluation")
G.add_edge("Irrelevant Sentences", "This sentence should not be considered relevant when answering questions.", relation = "example")
G.add_edge("Irrelevant Sentences", "Another irrelevant sentence is added here for the purpose of evaluation.", relation = "example")

# Настройка графа
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels = True, node_size = 3000, node_color = "lightblue", font_size = 8, font_weight = "bold")
edge_labels = nx.get_edge_attributes(G, 'relation')
nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, font_size = 8)

plt.title("Knowledge Graph for RAG")
plt.show()
