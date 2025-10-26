import os
import pickle
import gradio as gr
import numpy as np
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ===============================
# 1️⃣ DocumentStore
# ===============================
class DocumentStore:
    def __init__(self, storage_dir="debug/databases"):
        self.documents = {}
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def add_document(self, file_path):
        if not file_path:
            return self.list_documents()
        file_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        self.documents[file_name] = content
        return self.list_documents()

    def remove_document(self, name):
        if name in self.documents:
            del self.documents[name]
        return self.list_documents()

    def list_documents(self):
        if not self.documents:
            return "(База пуста)"
        return "\n".join([f"{name}: {len(text)} символов" for name, text in self.documents.items()])

    def save_database(self, name):
        path = os.path.join(self.storage_dir, name + ".pkl")
        with open(path, "wb") as f:
            pickle.dump(self.documents, f)
        return f"✅ База '{name}' сохранена."

    def load_database(self, name):
        path = os.path.join(self.storage_dir, name + ".pkl")
        if not os.path.exists(path):
            return f"❌ База '{name}' не найдена."
        with open(path, "rb") as f:
            self.documents = pickle.load(f)
        return f"✅ База '{name}' загружена."

    def get_all_texts(self):
        return list(self.documents.values())

    def get_document_names(self):
        return list(self.documents.keys())


# ===============================
# 2️⃣ QueryPreprocessor
# ===============================
class QueryPreprocessor:
    def __init__(self, lowercase=True, remove_punct=True):
        self.lowercase = lowercase
        self.remove_punct = remove_punct

    def preprocess(self, query: str):
        text = query.strip()
        if self.lowercase:
            text = text.lower()
        if self.remove_punct:
            import re
            text = re.sub(r"[^\w\s]", "", text)
        return text


# ===============================
# 3️⃣ DocumentSearcher
# ===============================
class DocumentSearcher:
    def __init__(self, store: DocumentStore):
        self.store = store
        self.embedding_provider = "tfidf"
        self.vectorizer = None
        self.doc_vectors = None
        self.model_st = None

    def set_embedding_provider(self, provider):
        """Устанавливает способ векторизации"""
        self.embedding_provider = provider
        return f"🔄 Переключено на {provider}"

    def build_index(self):
        texts = self.store.get_all_texts()
        if not texts:
            return "⚠️ В базе нет документов для индексации."

        if self.embedding_provider == "tfidf":
            try:
                self.vectorizer = TfidfVectorizer(stop_words="english")
                self.doc_vectors = self.vectorizer.fit_transform(texts).toarray()
            except Exception as e:
                return f"❌ Ошибка при построении TF-IDF индекса: {e}"
            return f"✅ TF-IDF индекс построен ({len(texts)} документов)."

        elif self.embedding_provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                self.model_st = SentenceTransformer("all-MiniLM-L6-v2")
                self.doc_vectors = self.model_st.encode(texts, convert_to_numpy=True)
            except Exception as e:
                return f"❌ Ошибка при получении эмбеддингов от sentence-transformers: {e}"
            
            return f"✅ Индекс построен через Sentence Transformers ({len(texts)} документов)."

        elif self.embedding_provider == "ollama":
            embeddings = []
            for text in texts:
                try:
                    resp = ollama.embeddings(model="nomic-embed-text", prompt=text)
                    embeddings.append(resp["embedding"])                
                except Exception as e:
                    return f"❌ Ошибка при получении эмбеддингов от Ollama: {e}"

            self.doc_vectors = np.array(embeddings)
            return f"✅ Индекс построен через Ollama Embeddings ({len(texts)} документов)."

        return "❌ Неизвестный провайдер эмбеддингов."

    def embed_query(self, query):
        if self.embedding_provider == "tfidf":
            return self.vectorizer.transform([query]).toarray()
        elif self.embedding_provider == "sentence-transformers":
            return self.model_st.encode([query], convert_to_numpy=True)
        elif self.embedding_provider == "ollama":
            resp = ollama.embeddings(model="nomic-embed-text", prompt=query)
            return np.array([resp["embedding"]])
        else:
            raise ValueError("Неизвестный провайдер эмбеддингов")

    def search(self, query: str, top_k=3):
        if self.doc_vectors is None:
            return "❌ Индекс не построен. Сначала нажми 'Построить индекс'."
        query_vec = self.embed_query(query)
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            doc_name = self.store.get_document_names()[idx]
            score = similarities[idx]
            snippet = self.store.documents[doc_name][:300].replace("\n", " ")
            results.append((doc_name, round(float(score), 3), snippet))
        return results


# ===============================
# 4️⃣ Gradio UI
# ===============================
store = DocumentStore()
preprocessor = QueryPreprocessor()
searcher = DocumentSearcher(store)

with gr.Blocks(title="RAG Document Search (Embeddings Switch)") as demo:
    gr.Markdown("## 🧠 Поиск по смыслу с выбором провайдера эмбеддингов")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Добавить документ (.txt)", file_types=[".txt"], type="filepath")
            add_btn = gr.Button("Добавить документ")

            provider = gr.Radio(
                ["tfidf", "sentence-transformers", "ollama"],
                label="Провайдер эмбеддингов",
                value="tfidf"
            )
            build_btn = gr.Button("Построить индекс")

            query_input = gr.Textbox(label="Введите запрос", placeholder="Например: как работает ollama.chat?")
            search_btn = gr.Button("Поиск")

        with gr.Column(scale=2):
            output_box = gr.Textbox(label="Результаты", lines=15)

    # --- Логика ---
    add_btn.click(lambda f: store.add_document(f), inputs=[file_input], outputs=[output_box])
    provider.change(lambda p: searcher.set_embedding_provider(p), inputs=[provider], outputs=[output_box])
    build_btn.click(lambda: searcher.build_index(), outputs=[output_box])

    def on_search(query):
        clean = preprocessor.preprocess(query)
        results = searcher.search(clean)
        if isinstance(results, str):
            return results
        text = "\n\n".join([f"📄 {name} ({score}):\n{snippet}" for name, score, snippet in results])
        return text

    search_btn.click(on_search, inputs=[query_input], outputs=[output_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
