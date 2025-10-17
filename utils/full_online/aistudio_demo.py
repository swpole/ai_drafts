"""
Set the API key as an environment variable:

Search for "Environment Variables" in the search bar.
Choose to modify System Settings. You may have to confirm you want to do this.
In the system settings dialog, click the button labeled Environment Variables.
Under either User variables (for the current user) or System variables (applies to all users who use the machine), click New...
Specify the variable name as GEMINI_API_KEY. Specify your Gemini API Key as the variable value.
Click OK to apply the changes.
Open a new terminal session (cmd or Powershell) to get the new variable.

"""

from google import genai
from google.genai import types
client = genai.Client()

config = types.GenerateContentConfig()

system_instruction = config.system_instruction
temperature = config.temperature
top_p = config.top_p
top_k = config.top_k
max_output_tokens = config.max_output_tokens

safety_settings = config.safety_settings
safety_settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),]

config = types.GenerateContentConfig(
    system_instruction=system_instruction,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    max_output_tokens=max_output_tokens,
    safety_settings=safety_settings,
)




response = client.models.generate_content(
    model="gemini-2.5-flash",
    config=config,
    contents="Explain how AI works in a few words",
)

print(response.text)