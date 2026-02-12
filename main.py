import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import FakeEmbeddings

# ===============================
# 1. Carregar dados
# ===============================
file_path = "inputs/dados.txt"

if not os.path.exists(file_path):
    raise FileNotFoundError("Arquivo dados.txt não encontrado na pasta inputs/")

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

print("Documento carregado com sucesso.")

# ===============================
# 2. Fragmentação (Chunking)
# ===============================
text_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

docs = text_splitter.split_text(text)

print(f"Documento dividido em {len(docs)} chunks.")

# ===============================
# 3. Criar Base Vetorial
# ===============================
embedding_model = FakeEmbeddings(size=1536)

vectorstore = Chroma.from_texts(
    texts=docs,
    embedding=embedding_model,
    persist_directory="./chroma_db"  # salva localmente
)

print("Base vetorial criada com sucesso.")

# ===============================
# 4. Busca Contextual
# ===============================
query = "Qual a vantagem do Docker?"

resultados = vectorstore.similarity_search(query, k=2)

print("\n==============================")
print(f"Pergunta: {query}")
print("==============================")

for i, doc in enumerate(resultados):
    print(f"\nResultado {i+1}:")
    print(doc.page_content)
