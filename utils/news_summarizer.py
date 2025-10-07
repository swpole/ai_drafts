import gradio as gr
import feedparser
import requests
import subprocess
from duckduckgo_search import DDGS
from newspaper import Article
from bs4 import BeautifulSoup

class NewsSummarizer:
    def __init__(self):
        self.titles_links_global = []
        self.RSS_FEEDS = {
            "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
            "Reuters": "http://feeds.reuters.com/reuters/topNews",
            "CNN": "http://rss.cnn.com/rss/edition.rss",
            "УНИАН": "https://rss.unian.net/site/news_ukr.rss"
        }

    # ---------- Поиск новостей ----------
    def search_duckduckgo(self, query, num=10):
        results = []
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=num):
                results.append({
                    "title": r.get("title"),
                    "link": r.get("url"),
                    "source": r.get("source"),
                    "date": r.get("date")
                })
        return results

    def search_newsapi(self, query, num=10, api_key="YOUR_API_KEY"):
        url = f"https://newsapi.org/v2/everything?q={query}&pageSize={num}&apiKey={api_key}"
        r = requests.get(url).json()
        results = []
        for article in r.get("articles", []):
            results.append({
                "title": article.get("title"),
                "link": article.get("url"),
                "source": article.get("source", {}).get("name"),
                "date": article.get("publishedAt")
            })
        return results

    def search_rss(self, num=10):
        results = []
        for name, url in self.RSS_FEEDS.items():
            feed = feedparser.parse(url)
            for entry in feed.entries[:num//len(self.RSS_FEEDS)]:
                results.append({
                    "title": entry.get("title"),
                    "link": entry.get("link"),
                    "source": name,
                    "date": entry.get("published", "")
                })
        return results

    def search_news(self, query, method):
        if method == "DuckDuckGo":
            return self.search_duckduckgo(query)
        elif method == "NewsAPI":
            return self.search_newsapi(query)
        elif method == "RSS":
            return self.search_rss()
        else:
            return [{"title": "Ошибка: неизвестный метод", "link": "", "source": "", "date": ""}]

    # ---------- Форматирование результатов ----------
    def format_results(self, results):
        if not results:
            self.titles_links_global = []
            return "<b>Нет результатов.</b>", []
        html = "<ol>"
        self.titles_links_global = []
        news_titles = []
        for r in results:
            title = r['title'] or "Без заголовка"
            source = r['source'] or "Неизвестный источник"
            date = r['date'] or ""
            link = r['link'] or "#"
            html += f"<li><a href='{link}' target='_blank'>{title}</a> <i>({source}, {date})</i></li>"
            self.titles_links_global.append((title, link))
            news_titles.append(title)
        html += "</ol>"
        return html, news_titles

    # ---------- Получение моделей Ollama ----------
    def get_ollama_models(self):
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            models = []
            for line in result.stdout.splitlines()[1:]:
                parts = line.split()
                if parts:
                    models.append(parts[0])
            return models if models else ["Нет моделей"]
        except Exception as e:
            return [f"Ошибка: {e}"]

    # ---------- Извлечение текста статьи ----------
    def extract_article_text(self, url):
        try:
            article = Article(
                url,
                language='ru',
                browser_user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                                    "Chrome/120.0.0.0 Safari/537.36"))
            article.download()
            article.parse()
            if article.text and len(article.text.split()) > 50:
                return article.text
        except Exception:
            pass

        try:
            headers = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                                      "Chrome/120.0.0.0 Safari/537.36")}
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            paragraphs = [p.get_text() for p in soup.find_all("p")]
            text = "\n".join(paragraphs).strip()

            if text and len(text.split()) > 30:
                return text
            else:
                return "Не удалось извлечь текст статьи (слишком мало текста)"
        except Exception as e:
            return f"Не удалось извлечь текст статьи: {e}"

    # ---------- Составление резюме ----------
    def summarize_news(self, news_text, prompt, llm_model):
        try:
            full_prompt = f"{prompt}\n\n{news_text}"
            result = subprocess.run(
                ["ollama", "run", llm_model],
                input=full_prompt,
                text=True,
                capture_output=True
            )
            return result.stdout.strip() if result.stdout else "Нет ответа от модели"
        except Exception as e:
            return f"Ошибка: {e}"

    # ---------- UI ----------
    def create_interface(self):
        with gr.Blocks() as interface:
            gr.Markdown("## 📰 Автоматический поиск и резюмирование новостей")

            with gr.Row():
                topic = gr.Textbox(label="Введите тему", value="Война в Украине")
                method = gr.Radio(["DuckDuckGo", "NewsAPI", "RSS"], value="DuckDuckGo", label="Метод поиска")

            search_btn = gr.Button("Поиск")
            output = gr.HTML(label="Результаты")
            
            news_dropdown = gr.Dropdown(choices=[], label="Выберите новость для резюмирования")
            llm_dropdown = gr.Dropdown(choices=self.get_ollama_models(), label="Выберите LLM (Ollama)", value=None)
            prompt_input = gr.Textbox(label="Промпт для составления резюме", 
                                    value="Составь краткое и понятное резюме новости для видео YouTube.", lines=3)
            
            extract_btn = gr.Button("Извлечь текст статьи")
            extract_output = gr.Textbox(label="Текст статьи", lines=10)
            
            summarize_btn = gr.Button("Составить резюме")
            self.summary_output = gr.Textbox(label="Резюме", lines=10, interactive=True)

            # ---------- Логика ----------
            def search_and_update_dropdown(query, method, dropdown):
                html, news_titles = self.format_results(self.search_news(query, method))
                return html, gr.update(choices=news_titles, value=news_titles[0] if news_titles else None)

            search_btn.click(fn=search_and_update_dropdown, 
                            inputs=[topic, method, news_dropdown], 
                            outputs=[output, news_dropdown])

            def extract_selected(news_choice, prompt, llm_model):
                if not news_choice or not self.titles_links_global:
                    return "Выберите новость и модель"
                link = None
                for title, lnk in self.titles_links_global:
                    if title == news_choice:
                        link = lnk
                        break
                if not link:
                    return "Ссылка для этой новости не найдена"
                news_text = self.extract_article_text(link)
                return news_text

            extract_btn.click(fn=extract_selected, 
                            inputs=[news_dropdown, prompt_input, llm_dropdown],
                            outputs=[extract_output])
            
            def summarize_text(news_text, prompt, llm_model):
                return self.summarize_news(news_text, prompt, llm_model)

            summarize_btn.click(fn=summarize_text, 
                                inputs=[extract_output, prompt_input, llm_dropdown],
                                outputs=[self.summary_output])    

        return interface


if __name__ == "__main__":
    news_summarizer = NewsSummarizer()
    interface = news_summarizer.create_interface()
    interface.launch()
