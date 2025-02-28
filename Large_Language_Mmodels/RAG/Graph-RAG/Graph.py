import networkx as nx
import matplotlib.pyplot as plt

# Создание графа
G = nx.Graph()

# Добавление узлов
G.add_node("Dr. Smith", type = "Scientist")
G.add_node("Quantum Computing", type = "Topic")
G.add_node("University of Science", type = "University")

# Добавление ребер
G.add_edge("Dr. Smith", "Quantum Computing", relation = "researches")
G.add_edge("Dr. Smith", "University of Science", relation = "works_at")

# Визуализация графа
pos = nx.spring_layout(G)  # Расположение узлов
nx.draw(G, pos, with_labels = True, node_size = 2000, node_color = "lightblue")
edge_labels = nx.get_edge_attributes(G, "relation")
nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)
plt.show()
