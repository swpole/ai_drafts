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

    def remove_document(self, file_name):
        if file_name in self.documents:
            del self.documents[file_name]
            return f"🗑 Документ '{file_name}' удалён."
        return f"⚠️ Документ '{file_name}' не найден."

    def list_documents(self):
        if not self.documents:
            return "(База пуста)"
        return "\n".join([f"{name}: {len(text)} символов" for name, text in self.documents.items()])

    def get_document_text(self, file_name, max_len=500):
        if file_name not in self.documents:
            return "⚠️ Документ не найден."
        text = self.documents[file_name]
        snippet = text[:max_len].replace("\n", " ")
        if len(text) > max_len:
            snippet += "..."
        return snippet

    def save_database(self, name):
        path = os.path.join(self.storage_dir, name + ".pkl")
        with open(path, "wb") as f:
            pickle.dump(self.documents, f)
        return f"✅ База '{name}' сохранена."

    def load_database(self, name):
        path = os.path.join(self.storage_dir, name + ".pkl")
        if not os.path.exists(path):
            return "❌ Файл не найден."
        with open(path, "rb") as f:
            self.documents = pickle.load(f)
        return f"✅ База '{name}' загружена ({len(self.documents)} документов)."

    def get_all_texts(self):
        return list(self.documents.values())

    def get_document_names(self):
        return list(self.documents.keys())


# ===============================
# 2️⃣ QueryPreprocessor
# ===============================
class QueryPreprocessor:
    def preprocess(self, query: str):
        text = query.strip().lower()
        import re
        text = re.sub(r"[^\w\s]", "", text)
        return text


# ===============================
# 3️⃣ DocumentSearcher
# ===============================
class DocumentSearcher:
    def __init__(self, store: DocumentStore, index_dir="debug/indexes"):
        self.store = store
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)

        self.embedding_provider = "tfidf"
        self.vectorizer = None
        self.model_st = None
        self.doc_vectors = None

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
            return f"✅ Sentence Transformers индекс ({len(texts)} документов)."

        elif self.embedding_provider == "ollama":
            embeddings = []
            for text in texts:
                resp = ollama.embeddings(model="nomic-embed-text", prompt=text)
                embeddings.append(resp["embedding"])
            self.doc_vectors = np.array(embeddings)
            return f"✅ Ollama Embeddings индекс ({len(texts)} документов)."

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

    def save_index(self, name):
        path = os.path.join(self.index_dir, name + ".pkl")
        data = {
            "provider": self.embedding_provider,
            "vectorizer": self.vectorizer,
            "doc_vectors": self.doc_vectors,
            "documents": self.store.documents,  # сохраняем документы
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        return f"✅ Индекс '{name}' сохранён."

    def load_index(self, name):
        path = os.path.join(self.index_dir, name + ".pkl")
        if not os.path.exists(path):
            return "❌ Индекс не найден."
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.embedding_provider = data["provider"]
        self.vectorizer = data["vectorizer"]
        self.doc_vectors = data["doc_vectors"]
        self.store.documents = data.get("documents", {})
        return f"✅ Индекс '{name}' загружен ({len(self.store.documents)} документов)."

    def list_indexes(self):
        files = [f[:-4] for f in os.listdir(self.index_dir) if f.endswith(".pkl")]
        return files or ["(нет индексов)"]


# ===============================
# 4️⃣ AnswerGenerator
# ===============================
class AnswerGenerator:
    def __init__(self):
        self.gemini_models = ["gemini-2.5-flash"]
        self.ollama_models = self._load_ollama_models()

    def _load_ollama_models(self):
        try:
            models = ollama.list()
            return [m["model"] for m in models["models"]]
        except Exception:
            return ["llama3", "mistral", "phi3"]
        
    def generate_answer_multiuser(self, llm_provider, model, query, context_texts, task_prompt="Ответь кратко и точно, опираясь только на контекст."):
        """
        Генерация ответа с учётом кастомного задания (task_prompt)
        """
        context = "\n\n".join([f"---\n{c}" for c in context_texts])
        prompt = f"Контекст:\n{context}\n\nВопрос: {query}\n\n{task_prompt}"

        if llm_provider == "ollama":
            try:
                resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
                return resp["message"]["content"].strip()
            except Exception as e:
                return f"[Ошибка Ollama] {e}"

        elif llm_provider == "gemini":
            try:
                from google import genai
                from google.genai import types

                client = genai.Client()
                config = types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=1024,
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                        )
                    ]
                )

                resp = client.models.generate_content(
                    model=model,
                    config=config,
                    contents=prompt,
                )
                return resp.text.strip()
            except Exception as e:
                return f"[Ошибка Gemini] {e}"

        return "[Ошибка] Неизвестный провайдер LLM."

    def generate_answer_full(self, provider: str, model: str, full_prompt: str) -> str:
        try:
            if provider == "ollama":
                stream = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Ты помощник, который отвечает чётко и по делу."},
                        {"role": "user", "content": full_prompt}
                    ],
                    stream=False,
                )
                return stream["message"]["content"].strip()

            elif provider == "gemini":
                try:
                    from google import genai
                    from google.genai import types
                    client = genai.Client()
                    resp = client.models.generate_content(
                        model=model,
                        contents=full_prompt,
                    )
                    return resp.text.strip()
                except Exception as e:
                    return f"[Ошибка Gemini] {e}"

            else:
                return f"❌ Неизвестный провайдер: {provider}"

        except Exception as e:
            return f"⚠️ Ошибка генерации: {e}"

    def generate_answer(self, llm_provider, model, query, context_texts):
        context = "\n\n".join([f"---\n{c}" for c in context_texts])
        prompt = f"Контекст:\n{context}\n\nВопрос: {query}\n\nОтветь кратко и точно, опираясь только на контекст."

        return self.generate_answer_full(llm_provider, model, prompt)


# ===============================
# 5️⃣ Gradio UI
# ===============================
store = DocumentStore()
preprocessor = QueryPreprocessor()
searcher = DocumentSearcher(store)
generator = AnswerGenerator()

with gr.Blocks(title="RAG v5 — управление документами и индексами") as demo:
    gr.Markdown("## 🧠 Retrieval-Augmented Generation (RAG v5)\nДобавляйте, удаляйте, просматривайте документы и стройте индексы.")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Добавить документ (.txt)", file_types=[".txt"], type="filepath")
            add_btn = gr.Button("Добавить документ")

            doc_dropdown = gr.Dropdown(label="Документы в базе", choices=store.get_document_names())
            remove_btn = gr.Button("🗑 Удалить выбранный документ")

            provider = gr.Radio(["tfidf", "sentence-transformers", "ollama"], label="Метод эмбеддингов", value="tfidf")
            build_btn = gr.Button("Построить индекс")

            index_name = gr.Textbox(label="Имя индекса для сохранения")
            save_index_btn = gr.Button("💾 Сохранить индекс")

            index_dropdown = gr.Dropdown(label="Загруженные индексы", choices=searcher.list_indexes())
            refresh_btn = gr.Button("🔄 Обновить список")
            load_index_btn = gr.Button("📂 Загрузить индекс")

            query_input = gr.Textbox(label="Запрос", value="Против каких компаний ввели новые экономические санкции?")
            search_btn = gr.Button("🔍 Поиск")

            llm_provider = gr.Radio(["ollama", "gemini"], label="Генератор ответа", value="ollama")
            ollama_models = gr.Dropdown(choices=generator.ollama_models, label="Модель Ollama", value=generator.ollama_models[0])
            gemini_models = gr.Dropdown(choices=generator.gemini_models, label="Модель Gemini", value=generator.gemini_models[0])

            answer_btn = gr.Button("💬 Составить ответ")

        with gr.Column(scale=2):
            doc_view = gr.Textbox(label="Содержимое выбранного документа", lines=10, interactive=False)
            output_box = gr.Textbox(label="Результаты", lines=20)

    # === Handlers ===
    def update_doc_list():
        return gr.update(choices=store.get_document_names())

    def on_doc_select(name):
        if not name:
            return ""
        return store.get_document_text(name)

    add_btn.click(
        lambda f: (store.add_document(f), update_doc_list()),
        inputs=[file_input],
        outputs=[output_box, doc_dropdown]
    )

    remove_btn.click(
        lambda name: (store.remove_document(name), update_doc_list()),
        inputs=[doc_dropdown],
        outputs=[output_box, doc_dropdown]
    )

    doc_dropdown.change(
        on_doc_select,
        inputs=[doc_dropdown],
        outputs=[doc_view]
    )

    provider.change(lambda p: searcher.set_embedding_provider(p), inputs=[provider], outputs=[output_box])
    build_btn.click(lambda: searcher.build_index(), outputs=[output_box])
    save_index_btn.click(lambda name: searcher.save_index(name), inputs=[index_name], outputs=[output_box])
    refresh_btn.click(lambda: gr.update(choices=searcher.list_indexes()), outputs=[index_dropdown])
    load_index_btn.click(
        lambda name: (searcher.load_index(name), update_doc_list()),
        inputs=[index_dropdown],
        outputs=[output_box, doc_dropdown]
    )

    def on_search(query):
        clean = preprocessor.preprocess(query)
        results = searcher.search(clean)
        if isinstance(results, str):
            return results
        return "\n\n".join([f"📄 {n} ({s}):\n{t}" for n, s, t in results])

    search_btn.click(on_search, inputs=[query_input], outputs=[output_box])

    def on_answer(query, llm_provider, ollama_model, gemini_model):
        results = searcher.search(query)
        if isinstance(results, str):
            return results
        context_texts = [snippet for _, _, snippet in results]
        model = ollama_model if llm_provider == "ollama" else gemini_model
        return generator.generate_answer(llm_provider, model, query, context_texts)

    answer_btn.click(on_answer, inputs=[query_input, llm_provider, ollama_models, gemini_models], outputs=[output_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
