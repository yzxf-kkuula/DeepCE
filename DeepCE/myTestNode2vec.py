import networkx as nx
from node2vec import Node2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt





G = nx.karate_club_graph()
#绘制图
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)  # 布局算法：让图结构更清晰
nx.draw(
    G,
    pos,
    with_labels=True,              # 显示节点编号
    node_color='skyblue',          # 节点颜色
    node_size=800,                 # 节点大小
    edge_color='gray',             # 边颜色
    font_size=10
)
plt.title("Zachary's Karate Club Graph")
plt.show()

# Node2Vec 生成嵌入向量（128维向量）
node2vec = Node2Vec(G, dimensions=128, walk_length=10, num_walks=100, workers=2)
model = node2vec.fit(window=5, min_count=1, batch_words=4)

# 获取所有节点的向量
node_ids = list(G.nodes())
node_embeddings = [model.wv[str(node)] for node in node_ids]

# 使用 TSNE 降维到 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
embeddings_2d = tsne.fit_transform(node_embeddings)

# 可视化嵌入
plt.figure(figsize=(10, 7))
for i, label in enumerate(node_ids):
    x, y = embeddings_2d[i]
    plt.scatter(x, y, color='blue')
    plt.text(x + 0.1, y + 0.1, str(label), fontsize=9)

plt.title('Node2Vec Embedding Visualization (TSNE)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()
