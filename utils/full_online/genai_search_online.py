from google import genai
from google.genai import types
import gradio as gr

class GeminiSearchOnline:
    def __init__(self, model_name='gemini-2.5-flash'):
        self.client = genai.Client()
        self.model_name = model_name

    def search(self, query):
        config = types.GenerateContentConfig(
            tools=[{"google_search": {}}],
            system_instruction="У тебя огромный опыт в поиске информации в интернете. Найди последние новости по предложенной теме и представь результаты в виде ссылок на страницы.",
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=query,
            config=config,
        )

        # Получаем ссылки из grounding_chunks
        #urls = [site.web.title for site in response.candidates[0].grounding_metadata.grounding_chunks]
        
        urls = [site.web.uri for site in response.candidates[0].grounding_metadata.grounding_chunks]

        # Формируем список для вывода
        if urls:
            html_links = "<br>".join([f'<a href="{url}" target="_blank">{url}</a>' for url in urls])
            return html_links
        else:
            return "No results found."

# Создаем объект класса
search_app = GeminiSearchOnline()

# Gradio интерфейс
iface = gr.Interface(
    fn=search_app.search,
    inputs=gr.Textbox(label="Enter your query", value="The war in der Ukraine"),
    outputs=gr.HTML(label="Search results"),
    title="Gemini Web Search",
    description="Search the web using Gemini-2.5 and get URLs."
)

iface.launch()
