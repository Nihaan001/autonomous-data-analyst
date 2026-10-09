import os
from dotenv import load_dotenv

load_dotenv(override=True)

key = os.getenv("GOOGLE_API_KEY")
model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

print("Folder      :", os.getcwd())
print("Key found   :", key is not None)
if not key:
    print("-> .env me GOOGLE_API_KEY nahi mili. .env isi folder me honi chahiye aur save honi chahiye.")
    raise SystemExit

print("Key length  :", len(key))
print("Starts with :", key[:3])
print("Space/quote :", (" " in key) or ('"' in key) or ("'" in key))
print("Model       :", model)

from langchain_google_genai import ChatGoogleGenerativeAI

try:
    reply = ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=0).invoke("Say hi in one word")
    content = reply.content
    if isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    print("RESULT      : SUCCESS ->", content)
except Exception as e:
    print("RESULT      : FAILED ->", str(e)[:400])
