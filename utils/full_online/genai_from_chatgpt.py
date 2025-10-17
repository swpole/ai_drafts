import os
import asyncio
import gradio as gr
from typing import List, Dict, Any, Optional

from google import genai
from google.genai import types
from google.genai.types import HttpOptions, Tool, ToolCodeExecution as CodeExecution

# ——— Клиент Gemini / GenAI ——
def make_client(use_vertex: bool = False, project: Optional[str] = None, location: Optional[str] = None):
    if use_vertex:
        client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=HttpOptions(api_version="v1")
        )
    else:
        client = genai.Client(
            http_options=HttpOptions(api_version="v1")
        )
    return client

# ——— Вспомогательная функция: строим список Content из истории + нового сообщения ——
def make_contents(history: List[Dict[str, str]], msg: str) -> List[types.Content]:
    contents: List[types.Content] = []
    for h in history:
        contents.append(types.Content(parts=[types.Part(text=h["content"])], role=h["role"]))
    contents.append(types.Content(parts=[types.Part(text=msg)], role="user"))
    return contents

# ——— Асинхронная генерация (не стриминг) ——
async def gemini_generate(
    client: genai.Client,
    model: str,
    contents: List[types.Content],
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    candidate_count: Optional[int],
    tools: Optional[List[Tool]]
) -> str:
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        candidate_count=candidate_count,
        tools=tools,
    )
    resp = await client.models.generate_content_async(
        model=model,
        contents=contents,
        config=config
    )
    # Предполагая, что .text даёт полный ответ
    return resp.text

# ——— Асинхронный стриминг: генератор, yielding части ответа ——
async def gemini_generate_stream(
    client: genai.Client,
    model: str,
    contents: List[types.Content],
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    candidate_count: Optional[int],
    tools: Optional[List[Tool]]
):
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        candidate_count=candidate_count,
        tools=tools,
    )
    stream = client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config
    )
    async for chunk in stream:
        # chunk может содержать .text или .parts — адаптируй под формат
        if hasattr(chunk, "text") and chunk.text is not None:
            yield chunk.text
        else:
            # Вариант: если chunk.parts содержит части
            parts = getattr(chunk, "parts", None)
            if parts:
                piece = "".join(p.text for p in parts if hasattr(p, "text"))
                yield piece

# ——— UI с Gradio ——
def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# Gemini Chat (новый API, стриминг)")

        with gr.Accordion("Основные параметры", open=True):
            use_vertex = gr.Checkbox(label="Use Vertex AI", value=False)
            project = gr.Textbox(label="Google Cloud Project (если Vertex)", placeholder="project-id")
            location = gr.Textbox(label="Location (если Vertex)", placeholder="global / region")
            model = gr.Dropdown(
                label="Model",
                choices=[
                    "gemini-2.5-flash",
                    "gemini-2.5-pro",
                    "gemini-2.5-flash-lite",
                    "gemini-2.0-flash",
                ],
                value="gemini-2.5-flash"
            )
            mode = gr.Radio(
                label="Режим",
                choices=["standard", "stream"],
                value="standard"
            )
            user_msg = gr.Textbox(label="Сообщение", placeholder="Введите текст…", lines=2)

        with gr.Accordion("Дополнительные настройки", open=False):
            temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.01, value=0.7)
            top_p = gr.Slider(label="Top-P", minimum=0.0, maximum=1.0, step=0.01, value=0.9)
            top_k = gr.Number(label="Top-K (int)", value=None)
            candidate_count = gr.Number(label="Candidate Count", value=None)
            enable_code_exec = gr.Checkbox(label="Enable code execution", value=False)

        chatbot = gr.Chatbot(label="Chat")
        state = gr.State([])

        send = gr.Button("Отправить")
        reset = gr.Button("Сбросить")

        async def on_send(
            use_vertex_, project_, location_, model_,
            mode_, user_msg_,
            temperature_, top_p_, top_k_, candidate_count_, enable_code_exec_,
            history
        ):

            client = make_client(use_vertex_, project_ or None, location_ or None)

            contents = make_contents(history, user_msg_)

            tools: Optional[List[Tool]] = None
            if enable_code_exec_:
                tools = [Tool(code_execution=CodeExecution(language="PYTHON"))]

            tk = int(top_k_) if (top_k_ is not None) else None
            cc = int(candidate_count_) if (candidate_count_ is not None) else None

            new_hist = history.copy()
            new_hist.append({"role": "user", "content": user_msg_})

            if mode_ == "standard":
                # просто ждем полного ответа
                resp_text = await gemini_generate(
                    client=client,
                    model=model_,
                    contents=contents,
                    temperature=temperature_,
                    top_p=top_p_,
                    top_k=tk,
                    candidate_count=cc,
                    tools=tools
                )
                new_hist.append({"role": "assistant", "content": resp_text})
                return new_hist, new_hist

            elif mode_ == "stream":
                # Асинхронный стриминг: возвращаем генератор, который yield’ит промежуточные состояния
                async def streamer():
                    partial = ""
                    async for chunk in gemini_generate_stream(
                        client=client,
                        model=model_,
                        contents=contents,
                        temperature=temperature_,
                        top_p=top_p_,
                        top_k=tk,
                        candidate_count=cc,
                        tools=tools
                    ):
                        partial += chunk
                        # каждую итерацию yield полный диалог до текущего момента
                        hist_temp = new_hist + [{"role": "assistant", "content": partial}]
                        # Преобразуем в формат, который Chatbot понимает: список список [ [user1, assistant1], [user2, assistant2], ... ]
                        yield [
                            [m["content"] for m in hist_temp]
                        ]
                    # После завершения стрима — можно завершить, уже не нужно return

                return streamer(), new_hist  # Gradio поймёт, что первый элемент — генератор

            else:
                # fallback как стандартный режим
                resp_text = await gemini_generate(
                    client=client,
                    model=model_,
                    contents=contents,
                    temperature=temperature_,
                    top_p=top_p_,
                    top_k=tk,
                    candidate_count=cc,
                    tools=tools
                )
                new_hist.append({"role": "assistant", "content": resp_text})
                return new_hist, new_hist

        send.click(
            on_send,
            inputs=[
                use_vertex, project, location, model,
                mode, user_msg,
                temperature, top_p, top_k, candidate_count, enable_code_exec,
                state
            ],
            outputs=[chatbot, state]
        )

        reset.click(lambda: ([], []), None, [chatbot, state])

    demo.queue()  # включаем очередь (нужно для стриминга) :contentReference[oaicite:1]{index=1}
    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)
