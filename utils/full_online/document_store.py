import gradio as gr
import os
import pickle

class DocumentStore:
    def __init__(self, storage_dir="debug/databases"):
        self.documents = {}  # ключ: имя файла, значение: текст
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    # --- Добавление документа ---
    def add_document(self, file_path):
        if not file_path:
            return self.list_documents(), gr.update(choices=list(self.documents.keys()))

        file_name = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            self.documents[file_name] = content
        except Exception as e:
            return f"❌ Ошибка при чтении файла: {e}", gr.update(choices=list(self.documents.keys()))

        return self.list_documents(), gr.update(choices=list(self.documents.keys()), value=file_name)

    # --- Удаление документа ---
    def remove_document(self, file_name):
        if file_name in self.documents:
            del self.documents[file_name]
        return self.list_documents(), gr.update(choices=list(self.documents.keys()), value=None)

    # --- Список документов / содержание базы ---
    def list_documents(self):
        if not self.documents:
            return "(База пуста)"
        return "\n".join([f"{name}: {len(text)} символов" for name, text in self.documents.items()])

    # --- Сохранение базы на диск ---
    def save_database(self, db_name):
        if not db_name:
            return "⚠️ Укажите имя базы.", gr.update()
        path = os.path.join(self.storage_dir, db_name + ".pkl")
        with open(path, "wb") as f:
            pickle.dump(self.documents, f)
        return f"✅ База '{db_name}' сохранена.", gr.update(choices=self.list_saved_databases(), value=db_name)

    # --- Загрузка базы с диска ---
    def load_database(self, db_name):
        path = os.path.join(self.storage_dir, db_name + ".pkl")
        if not os.path.exists(path):
            return f"❌ База '{db_name}' не найдена.", gr.update()
        with open(path, "rb") as f:
            self.documents = pickle.load(f)
        return f"✅ База '{db_name}' загружена.\n{self.list_documents()}", gr.update(choices=list(self.documents.keys()))

    # --- Список сохранённых баз ---
    def list_saved_databases(self):
        files = os.listdir(self.storage_dir)
        return [f.replace(".pkl", "") for f in files if f.endswith(".pkl")]


# --- Инициализация хранилища ---
store = DocumentStore()

# --- Интерфейс Gradio ---
with gr.Blocks(title="RAG Document Store") as demo:
    gr.Markdown("## 📚 Хранилище документов RAG")

    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(
                label="Добавить документ (.txt)",
                file_types=[".txt"],
                type="filepath"  # ✅ путь к файлу вместо байтов
            )
            add_btn = gr.Button("Добавить")
            remove_doc = gr.Dropdown(label="Удалить документ", choices=[], interactive=True)
            remove_btn = gr.Button("Удалить выбранный документ")

            output_display = gr.Textbox(label="Содержимое базы", interactive=False, lines=10)

        with gr.Column(scale=1):
            db_name_input = gr.Textbox(label="Имя базы для сохранения / загрузки")
            save_btn = gr.Button("💾 Сохранить базу")
            saved_dbs = gr.Dropdown(choices=store.list_saved_databases(), label="📂 Выбрать сохранённую базу")
            load_btn = gr.Button("📥 Загрузить выбранную базу")
            refresh_btn = gr.Button("🔄 Обновить список баз")

    # --- Обработчики ---
    add_btn.click(store.add_document, inputs=[file_input], outputs=[output_display, remove_doc])
    remove_btn.click(store.remove_document, inputs=[remove_doc], outputs=[output_display, remove_doc])
    save_btn.click(store.save_database, inputs=[db_name_input], outputs=[output_display, saved_dbs])
    load_btn.click(store.load_database, inputs=[saved_dbs], outputs=[output_display, remove_doc])
    refresh_btn.click(lambda: gr.update(choices=store.list_saved_databases()), outputs=[saved_dbs])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
