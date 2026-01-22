from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.agent_core import build_agent, load_data_if_available

app = FastAPI()
AGENT = build_agent()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    return {"status": "ok"}


@app.on_event("startup")
def startup_load() -> None:
    try:
        load_data_if_available(AGENT)
    except Exception as exc:
        print(f"Startup data load failed: {exc}")


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    answer = AGENT(req.question)
    return {"answer": str(answer)}
