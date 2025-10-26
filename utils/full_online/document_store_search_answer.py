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
    def __init__(self, storage_dir="databases"):
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

    def list_documents(self):
        if not self.documents:
            return "(База пуста)"
        return "\n".join([f"{name}: {len(text)} символов" for name, text in self.documents.items()])

    def save_database(self, name):
        path = os.path.join(self.storage_dir, name + ".pkl")
        with open(path, "wb") as f:
            pickle.dump(self.documents, f)
        return f"✅ База '{name}' сохранена."

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
        self.embedding_provider = provider
        return f"🔄 Переключено на {provider}"

    def build_index(self):
        texts = self.store.get_all_texts()
        if not texts:
            return "⚠️ В базе нет документов для индексации."

        if self.embedding_provider == "tfidf":
            self.vectorizer = TfidfVectorizer(stop_words="russian")
            self.doc_vectors = self.vectorizer.fit_transform(texts).toarray()
            return f"✅ TF-IDF индекс построен ({len(texts)} документов)."

        elif self.embedding_provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                self.model_st = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                return "❌ Установи пакет sentence-transformers."
            self.doc_vectors = self.model_st.encode(texts, convert_to_numpy=True)
            return f"✅ Индекс через Sentence Transformers ({len(texts)} документов)."

        elif self.embedding_provider == "ollama":
            embeddings = []
            for text in texts:
                resp = ollama.embeddings(model="nomic-embed-text", prompt=text)
                embeddings.append(resp["embedding"])
            self.doc_vectors = np.array(embeddings)
            return f"✅ Индекс через Ollama Embeddings ({len(texts)} документов)."

    def embed_query(self, query):
        if self.embedding_provider == "tfidf":
            return self.vectorizer.transform([query]).toarray()
        elif self.embedding_provider == "sentence-transformers":
            return self.model_st.encode([query], convert_to_numpy=True)
        elif self.embedding_provider == "ollama":
            resp = ollama.embeddings(model="nomic-embed-text", prompt=query)
            return np.array([resp["embedding"]])

    def search(self, query: str, top_k=3):
        if self.doc_vectors is None:
            return "❌ Индекс не построен."
        query_vec = self.embed_query(query)
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            doc_name = self.store.get_document_names()[idx]
            score = similarities[idx]
            snippet = self.store.documents[doc_name][:500].replace("\n", " ")
            results.append((doc_name, round(float(score), 3), snippet))
        return results


# ===============================
# 4️⃣ AnswerGenerator
# ===============================
class AnswerGenerator:
    def __init__(self):
        self.gemini_models = ["gemini-2.0-flash", "gemini-2.0-pro"]
        self.ollama_models = self._load_ollama_models()

    def _load_ollama_models(self):
        try:
            models = ollama.list()
            return [m["model"] for m in models["models"]]
        except Exception:
            return ["llama3", "mistral", "phi3"]

    def generate_answer(self, llm_provider, model, query, context_texts):
        context = "\n\n".join([f"---\n{c}" for c in context_texts])
        prompt = f"Контекст:\n{context}\n\nВопрос: {query}\n\nДай краткий и точный ответ, опираясь только на контекст."

        if llm_provider == "ollama":
            try:
                resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
                return resp["message"]["content"].strip()
            except Exception as e:
                return f"[Ошибка Ollama] {e}"

        elif llm_provider == "gemini":
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                model_obj = genai.GenerativeModel(model)
                resp = model_obj.generate_content(prompt)
                return resp.text.strip()
            except Exception as e:
                return f"[Ошибка Gemini] {e}"

        return "[Ошибка] Неизвестный провайдер LLM."


# ===============================
# 5️⃣ Gradio UI
# ===============================
store = DocumentStore()
preprocessor = QueryPreprocessor()
searcher = DocumentSearcher(store)
generator = AnswerGenerator()

with gr.Blocks(title="RAG v3 — Semantic Search + Answer Generation") as demo:
    gr.Markdown("## 🤖 Retrieval-Augmented Generation (RAG)\nДобавьте документы → Постройте индекс → Найдите → Составьте ответ")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Добавить документ (.txt)", file_types=[".txt"], type="filepath")
            add_btn = gr.Button("Добавить")

            provider = gr.Radio(["tfidf", "sentence-transformers", "ollama"], label="Провайдер эмбеддингов", value="tfidf")
            build_btn = gr.Button("Построить индекс")

            query_input = gr.Textbox(label="Запрос")
            search_btn = gr.Button("🔍 Поиск")

            llm_provider = gr.Radio(["ollama", "gemini"], label="Генератор ответа", value="ollama")
            ollama_models = gr.Dropdown(choices=generator.ollama_models, label="Модель Ollama", value=generator.ollama_models[0])
            gemini_models = gr.Dropdown(choices=generator.gemini_models, label="Модель Gemini", value=generator.gemini_models[0])

            answer_btn = gr.Button("💬 Составить ответ")

        with gr.Column(scale=2):
            search_results = gr.Textbox(label="Результаты поиска", lines=10)
            final_answer = gr.Textbox(label="Ответ", lines=10)

    # --- Handlers ---
    add_btn.click(lambda f: store.add_document(f), inputs=[file_input], outputs=[search_results])
    provider.change(lambda p: searcher.set_embedding_provider(p), inputs=[provider], outputs=[search_results])
    build_btn.click(lambda: searcher.build_index(), outputs=[search_results])

    def on_search(query):
        clean = preprocessor.preprocess(query)
        results = searcher.search(clean)
        if isinstance(results, str):
            return results
        text = "\n\n".join([f"📄 {name} ({score}):\n{snippet}" for name, score, snippet in results])
        return text

    search_btn.click(on_search, inputs=[query_input], outputs=[search_results])

    def on_answer(query, llm_provider, ollama_model, gemini_model):
        results = searcher.search(query)
        if isinstance(results, str):
            return results
        context_texts = [snippet for _, _, snippet in results]
        model = ollama_model if llm_provider == "ollama" else gemini_model
        answer = generator.generate_answer(llm_provider, model, query, context_texts)
        return answer

    answer_btn.click(
        on_answer,
        inputs=[query_input, llm_provider, ollama_models, gemini_models],
        outputs=[final_answer]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
