from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent_core import build_agent, load_data_if_available

app = FastAPI()
AGENT = build_agent()

@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers.setdefault(
        "Access-Control-Allow-Origin",
        "https://main.d13s0xvafu5qv7.amplifyapp.com",
    )
    response.headers.setdefault(
        "Access-Control-Allow-Methods",
        "GET,POST,OPTIONS",
    )
    response.headers.setdefault(
        "Access-Control-Allow-Headers",
        "Content-Type,Authorization",
    )
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://main.d13s0xvafu5qv7.amplifyapp.com"],
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
    prompt = f"{req.question.strip()} (reply in a conversational manner)"
    answer = AGENT(prompt)
    return {"answer": str(answer)}


@app.options("/ask")
def ask_options() -> dict:
    return {"status": "ok"}
