import gradio as gr
from document_store_search_answer_saveindex_removedoc import DocumentStore, QueryPreprocessor, DocumentSearcher, AnswerGenerator


class RAGChatUI:
    def __init__(self):
        self.store = DocumentStore()
        self.preprocessor = QueryPreprocessor()
        self.searcher = DocumentSearcher(self.store)
        self.generator = AnswerGenerator()

    # === Логика ===
    def refresh_indexes(self):
        return gr.update(choices=self.searcher.list_indexes())

    def load_index(self, index_name):
        msg = self.searcher.load_index(index_name)
        provider = self.searcher.embedding_provider
        return f"{msg}\n\n🔹 Метод эмбеддингов: {provider}"

    def answer_query(
        self,
        message,
        history,
        llm_provider,
        ollama_model,
        gemini_model,
        task_prompt,
    ):
        """
        Chat-RAG с историей, контекстом и пользовательским заданием (инструкцией).
        """
        if not message.strip():
            return "❌ Введите сообщение."

        # История диалога
        history_text = "\n".join(
            [f"{'Пользователь' if role == 'user' else 'Ассистент'}: {content}"
             for role, content in history]
        )

        # Поиск по базе
        clean_query = self.preprocessor.preprocess(message)
        search_results = self.searcher.search(clean_query)
        if isinstance(search_results, str):
            return search_results

        # Контекст из найденных документов
        context_texts = [snippet for _, _, snippet in search_results]
        context = "\n\n".join(context_texts)

        # Формирование промпта
        full_prompt = (
            "Ты — умный ассистент, использующий найденные документы.\n"
            "Следуй заданию ниже.\n\n"
            f"Задание: {task_prompt}\n\n"
            f"История диалога:\n{history_text}\n\n"
            f"Контекст из документов:\n{context}\n\n"
            f"Новый запрос пользователя: {message}\n\n"
            "Ответ:"
        )

        # Выбор модели
        model = ollama_model if llm_provider == "ollama" else gemini_model

        # Генерация
        answer = self.generator.generate_answer_full(
            llm_provider,
            model,
            full_prompt,        # теперь передаём полный промпт
        )

        # Отображаем также список найденных документов (для наглядности)
        doc_info = "\n".join([f"📄 {n} ({s})" for n, s, _ in search_results])
        return f"{answer}\n\n\n🔍 Использованы документы:\n{doc_info}"

    # === UI ===
    def create_ui(self):
        with gr.Blocks(title="RAG Chat Interface") as ui:
            gr.Markdown("## 💬 Chat with RAG System\nChat + Retrieval + Instruction = Powerful Contextual Assistant")

            with gr.Row():
                with gr.Column(scale=1):
                    index_dropdown = gr.Dropdown(
                        label="📂 Выберите индекс",
                        choices=self.searcher.list_indexes()
                    )
                    refresh_btn = gr.Button("🔄 Обновить список индексов")
                    load_msg = gr.Textbox(label="Статус загрузки", interactive=False)

                    task_prompt = gr.Textbox(
                        label="🧩 Задание для модели",
                        value="Дай развёрнутый и понятный ответ, основываясь только на приведённом контексте. Приводи все ключевые детали, чтобы ответ был максимально информативным и полезным, но не выходи за рамки контекста.",
                        lines=3
                    )

                    llm_provider = gr.Radio(
                        ["ollama", "gemini"],
                        label="Провайдер модели",
                        value="ollama"
                    )
                    ollama_models = gr.Dropdown(
                        choices=self.generator.ollama_models,
                        label="Модель Ollama",
                        value=self.generator.ollama_models[0]
                    )
                    gemini_models = gr.Dropdown(
                        choices=self.generator.gemini_models,
                        label="Модель Gemini",
                        value=self.generator.gemini_models[0]
                    )

                with gr.Column(scale=3):
                    chat = gr.ChatInterface(
                        fn=self.answer_query,
                        additional_inputs=[
                            llm_provider,
                            ollama_models,
                            gemini_models,
                            task_prompt,
                        ],
                        title="🧠 RAG Assistant",
                        description="Ассистент, который использует документы и задание для ответов.",
                    )

            # === Handlers ===
            refresh_btn.click(self.refresh_indexes, outputs=[index_dropdown])
            index_dropdown.change(self.load_index, inputs=[index_dropdown], outputs=[load_msg])

        return ui


if __name__ == "__main__":
    app = RAGChatUI()
    ui = app.create_ui()
    ui.launch(server_name="0.0.0.0", share=False)
