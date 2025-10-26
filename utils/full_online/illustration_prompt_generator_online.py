# Geterate Illustration Prompts
# intut text - long text
# output -  prompts

from llm_interface_online import LLMInterfaceOnline
import gradio as gr
import re

class IllustrationPromptGeneratorOnline:
    def __init__(self):
        self.input_text = """
Жил-был старик со старухою. Просит старик:
- Испеки, старуха, колобок.
- Из чего печь-то? Муки нету.
- Э-эх , старуха! По коробу поскреби, по сусеку помети, авось и муки наберётся.
Взяла старуха крылышко, по коробу поскребла, по сусеку помела, и набралось муки пригоршни с две. 
Замесила на сметане, изжарила в масле и положила на окошко постудить.
Колобок полежал-полежал, да вдруг и покатился. С окна на лавку, с лавки на пол, по полу да к дверям, перепрыгнул через порог в сени, из сеней на крыльцо, с крыльца на двор, со двора за ворота, дальше и дальше.
Катится, катится колобок по дороге, а навстречу ему заяц:
- Колобок, колобок! Я тебя съем!
- Не ешь меня, косой зайчик! Я тебе песенку спою,- сказал колобок и запел:

Я колобок, колобок
По коробу скребён,
По сусеку метён,
На сметане мешон,
Да в масле пряжон,
На окошке стужон;

Я от дедушки ушёл,
Я от бабушки ушёл,
И от тебя, зайца, и не хитро уйти!

И покатился себе дальше, только заяц его и видел!.
Катится колобок, а навстречу ему волк:
- Колобок, колобок! Я тебя съем!
- Не ешь меня, серый волк! Я тебе песенку спою! И запел:

Я колобок, колобок
По коробу скребён,
По сусеку метён,
На сметане мешон,
Да в масле пряжон,
На окошке стужон;

Я от дедушки ушёл,
Я от бабушки ушёл,
Я от зайца ушёл,
От тебя, волка, не хитро уйти!

И покатился себе дальше, только волк его и видел!..
Катится колобок, а навстречу ему медведь:
- Колобок, колобок! Я тебя съем.
- Где тебе, косолапому, съесть меня! И запел:

Я колобок, колобок
По коробу скребён,
По сусеку метён,
На сметане мешон,
Да в масле пряжон,
На окошке стужон;

Я от дедушки ушёл,
Я от бабушки ушёл,
Я от зайца ушёл,
Я от волка ушёл,
От тебя, медведь, и подавно уйду!

И опять укатился, только медведь его и видел!..
Катится, катится колобок, а навстречу ему лиса:
- Здравствуй, колобок! Какой ты хорошенький! А колобок запел:

Я колобок, колобок
По коробу скребён,
По сусеку метён,
На сметане мешон,
Да в масле пряжон,
На окошке стужон;

Я от дедушки ушёл,
Я от бабушки ушёл,
Я от зайца ушёл,
Я от волка ушёл,
От медведя ушёл,
От тебя, лиса, и подавно уйду!
- Какая славная песенка! - сказала лиса. 
- Но ведь я, колобок, стара стала, плохо слышу. Сядь-ка на мою мордочку, да пропой ещё разок погромче.
Колобок вскочил лисе на мордочку и запел ту же песню.
- Какая хорошая песенка, колобок! Славная песенка, еще бы послушала! Сядь-ка на мой язычок да пропой в последний разок, - сказала лиса и высунула свой язык.
Колобок сдуру прыг ей на язык, а лиса -ам его! - и съела.
"""

        self.input_text = """Resume: The Big Bang is a physical theory that describes how the universe expanded from an initial state of high density and temperature. 
Various cosmological models based on the Big Bang concept explain a broad range of phenomena, including the abundance of light elements, the cosmic microwave background (CMB) radiation, and large-scale structure. 
The uniformity of the universe is explained through cosmic inflation, which is considered the age of the universe. 
Inspired by the laws of physics, models of the Big Bang describe an extraordinarily hot and dense primordial universe. 
Physics lacks a widely accpeted theory that can model the earliest conditions of the Big Bang, but evidence suggests it may occur in the era of matter-antimatter asymmetry, dark matter, and the origin of dark energy."""

        self.output_prompts = """Cinematic Prompt 1 (text): **Scene 1**
- Original text: The Big Bang is a physical theory that describes how the universe expanded from an initial state of high density and temperature. 
- Visual prompt with elements:
  • Frame composition (close-up, wide shot, etc.)
  • Lighting and atmosphere
  • Camera angle and perspective
  • Artistic style: realism
  • Emotional impact of the scene

Cinematic Prompt 2 (text): **Scene 2**
- Original text: The Big Bang is a physical theory that describes how the universe expanded from an initial state of high density and temperature. 
- Visual prompt with elements:
  • Frame composition (close-up, wide shot, etc.)
  • Lighting and atmosphere
  • Camera angle and perspective
  • Artistic style: realism
  • Emotional impact of the scene

Response format:
1. Take a break from the original text and select another text to prompt for visual description. 2. Choose a visual prompt based on the new text, and create a description ready for image generation. 3. Implement the cinematic prompts into the generated image.
"""
        
        self.scenes = {}

        self.prompts = ["""  • Frame composition (close-up, wide shot, etc.)
  • Lighting and atmosphere
  • Camera angle and perspective
  • Artistic style: realism
  • Emotional impact of the scene""", """  • Frame composition (close-up, wide shot, etc.)
  • Lighting and atmosphere
  • Camera angle and perspective
  • Artistic style: realism
  • Emotional impact of the scene"""]

        self.base_prompts = {
            "Детализированный": """Ты — эксперт по визуализации текста и созданию иллюстрационных промптов.
Твоя задача — превратить длинный текст в серию промптов для генерации изображений.

Правила работы:
1. Разбей текст на последовательные сегменты, каждый примерно по 4 секунды чтения (≈10 слов).
2. Для каждого сегмента:
   - Выведи сам текст сегмента.
   - Составь иллюстрационный промпт, включающий:
     • описание сцены, персонажей и их действий;
     • атмосферу и настроение (таинственно, радостно, тревожно и т.д.);
     • художественный стиль: {style};
     • детали окружения и фона;
     • цветовую гамму и композицию (если уместно).

Формат ответа:

**Сегмент 1 (текст)**
- Иллюстрационный промпт: …

**Сегмент 2 (текст)**
- Иллюстрационный промпт: …

…и так далее, пока не будет обработан весь текст.

Итог должен быть списком промптов для генерации иллюстраций, которые можно подавать напрямую в графическую модель. 
Для совместимости с генераторами изображений описания должны быть на английском языке""",

            "Кинематографический": """Ты — режиссер и художник-раскадровщик. Преврати текст в кинематографические промпты для генерации изображений.

Правила работы:
1. Разбей текст на сцены (каждая ≈50-60 слов).
2. Для каждой сцены укажи:
   - Текст оригинала
   - Визуальный промпт с элементами:
     • Композиция кадра (крупный план, общий план и т.д.)
     • Освещение и атмосфера
     • Ракурс и угол съемки
     • Художественный стиль: {style}
     • Эмоциональная нагрузка сцены

Формат ответа:

**Сцена 1 (текст)**
- Кинематографический промпт: …

**Сцена 2 (текст)**
- Кинематографический промпт: …

Создай визуально насыщенные описания, готовые для генерации изображений. 
Для совместимости с генераторами изображений описания должны быть на английском языке""",

            "Cinematic (en)": """You are a director and storyboard artist. Transform text into cinematic prompts for image generation.

Working rules:
1. Break the text into scenes (each ≈50-60 words).
2. For each scene specify:
   - Original text
   - Visual prompt with elements:
     • Frame composition (close-up, wide shot, etc.)
     • Lighting and atmosphere
     • Camera angle and perspective
     • Artistic style: {style}
     • Emotional impact of the scene

Response format:

**Scene 1 (text)**
- Cinematic prompt: …

**Scene 2 (text)**
- Cinematic prompt: …

Create visually rich descriptions ready for image generation.""",

            "Detailed (en)": """You are an expert in text visualization and creating illustration prompts.
Your task is to transform long text into a series of image generation prompts.

Working rules:
1. Divide the text into consecutive segments, each about 4 seconds of reading (≈50–60 words).
2. For each segment:
   - Output the segment text itself.
   - Create an illustration prompt including:
     • description of the scene, characters and their actions;
     • atmosphere and mood (mysterious, joyful, anxious, etc.);
     • artistic style: {style};
     • environment and background details;
     • color palette and composition (where appropriate).

Response format:

**Segment 1 (text)**
- Illustration prompt: …

**Segment 2 (text)**
- Illustration prompt: …

…and so on until the entire text is processed.

The result should be a list of prompts for illustration generation that can be fed directly into a graphics model."""
        }
        
        self.styles_ru = [
            "реализм",
            "книжная иллюстрация",
            "цифровая живопись",
            "акварель",
            "комикс",
            "минимализм",
            "стилизованная графика"
        ]
        
        self.styles_en = [
            "realism",
            "book illustration", 
            "digital painting",
            "watercolor",
            "comic",
            "minimalism",
            "stylized graphics"
        ]

        self.prompt_types = list(self.base_prompts.keys())
        
        # Определяем язык для каждого типа промпта
        self.prompt_languages = {
            "Детализированный": "ru",
            "Кинематографический": "ru", 
            "Cinematic (en)": "en",
            "Detailed (en)": "en"
        }

        self.create_interface()
        

    def get_styles_for_prompt_type(self, prompt_type):
        """Получение списка стилей в зависимости от типа промпта"""
        language = self.prompt_languages.get(prompt_type, "ru")
        if language == "en":
            return self.styles_en
        else:
            return self.styles_ru

    def get_default_style(self, prompt_type):
        """Получение стиля по умолчанию в зависимости от типа промпта"""
        language = self.prompt_languages.get(prompt_type, "ru")
        if language == "en":
            return "realism"
        else:
            return "реализм"

    def update_prompt(self, prompt_type, style):
        """Обновление промпта при смене типа промпта и стиля"""
        base_prompt = self.base_prompts.get(prompt_type, self.base_prompts["Детализированный"])
        return base_prompt.format(style=style)

    def update_interface(self, prompt_type):
        """Обновление интерфейса при смене типа промпта"""
        styles = self.get_styles_for_prompt_type(prompt_type)
        default_style = self.get_default_style(prompt_type)
        updated_prompt = self.update_prompt(prompt_type, default_style)
        
        return (
            gr.update(choices=styles, value=default_style),  # style_dropdown
            updated_prompt  # prompt_box
        )

    def split_scenes(self, input_text):
        """Разделение текста на сцены и формирование сегментов"""
        self.segments = []
        self.titles = []
        current_segment = ""

        all_patterns = r"\*\*Сегмент|\*\*Сцена|\*\*Segment|\*\*Scene"
        
        for line in input_text.split("\n"):
            if re.search(all_patterns, line, re.IGNORECASE):
                if current_segment:
                    self.segments.append(current_segment.strip())
                current_segment = line + "\n"
                self.titles.append(line)
            else:
                current_segment += line + "\n"
                
        if current_segment:
            self.segments.append(current_segment.strip())

        for i in range(len(self.segments)):
            text_lower = self.segments[i].lower()
            substring_lower = "Illustration Prompt".lower()
            index = text_lower.find(substring_lower)
            if index != -1:
                index+=len(substring_lower)
                splited=self.segments[i][index:]
                splited=splited.replace("\n", "")
                splited=splited.replace("*", "")
                splited=splited.replace(":", "")
                self.segments[i] = splited
            else:
                substring_lower = "Иллюстрационный промпт".lower()
                index = text_lower.find(substring_lower)
                if index != -1:
                    index+=len(substring_lower)
                    splited=self.segments[i][index:]
                    splited=splited.replace("\n", "")
                    splited=splited.replace("*", "")
                    splited=splited.replace(":", "")
                    self.segments[i] = splited
                else:
                    pass


        self.scenes=self.titles

        return gr.update(choices=self.scenes, value=self.scenes[0] if self.scenes else None)

    def show_scene_prompt(self, selected_title):
        """Отображение промпта выбранной сцены"""
        if not selected_title or not self.segments:
            return ""
        
        # Находим индекс сегмента, независимо от языка
        match = re.search(r"(Сегмент|Сцена|Segment|Scene)\s*(\d+)", selected_title, re.IGNORECASE)
        if not match:
            return ""
        index = int(match.group(2)) - 1
        if 0 <= index < len(self.segments):
            segment_text = self.segments[index]
            lines = segment_text.split("\n")
            
            # Ищем строку с промптом (разные варианты на разных языках)
            prompt_keywords = [
                "Иллюстрационный промпт", "Кинематографический промпт", 
                "Illustration prompt", "Cinematic prompt", "промпт:"
            ]
            for i, line in enumerate(lines):
                if any(keyword in line for keyword in prompt_keywords):
                    return "\n".join(lines[i:]).strip()
        
        return segment_text if 0 <= index < len(self.segments) else ""

    def create_interface(self):
        """Создание и запуск Gradio интерфейса"""

        self.llm_interface = LLMInterfaceOnline(
                title="Генератор иллюстрационных промптов",
                heading="Генерация промптов для иллюстраций",
                prompt_label="📌 Системный промпт",
                prompt_default=self.base_prompts["Детализированный"].format(style="реализм"),
                input_label="Исходный текст",
                input_placeholder="Вставьте сюда длинный текст...",
                input_value=self.input_text.strip(),
                generate_button_text="✨ Сгенерировать промпты",
                output_label="Список промптов для иллюстраций",
                typical_prompts=self.base_prompts,
                prompt_params={"Детализированный":self.styles_ru, 
                                "Кинематографический":self.styles_ru,
                                "Cinematic (en)":self.styles_en,
                                "Detailed (en)":self.styles_en,
                                },
                default_prompt_index=0, default_param_index=0
            )
            
        split_btn = gr.Button("Разделить текст на сегменты")
        self.scene_dropdown = gr.Dropdown(choices=self.scenes, label="Выберите сцену")       
        split_btn.click(
            fn=self.split_scenes,
            inputs=self.llm_interface.output_box.textbox,
            outputs=[self.scene_dropdown]
        )

        
        self.scene_prompt_box = gr.Textbox(
            label="Промпт выбранной сцены", 
            lines=5,
            max_lines=5,
            interactive=True,
            show_copy_button=True
        )


        # Отображение промпта выбранной сцены
        self.scene_dropdown.change(
            fn=self.show_scene_prompt,
            inputs=[self.scene_dropdown],
            outputs=self.scene_prompt_box
        )

        return

if __name__ == "__main__":
    with gr.Blocks() as interface:
        generator = IllustrationPromptGeneratorOnline()
    interface.launch()