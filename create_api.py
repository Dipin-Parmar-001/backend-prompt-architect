import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph.types import Command
from new_pipline import app
from fastapi import HTTPException

api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"]
)

class QueryRequest(BaseModel):
    query: str
    complexity: str
    thread_id: str

@api.post("/stream")
async def stream_pipline(request: QueryRequest):

    config = {"configurable": {"thread_id": request.thread_id}}

    init_state = {
        "user_query": request.query,
        "iteration_count": 0,
        "revision_count": [],
        "mcq_answer": {},
        "complexity": request.complexity,
        "target_model": "gemini-3.1"
    }

    async def event_generator():
        """
        This generator iterates through the graph's events.
        It sends strings formatted as 'data: {json}\n\n' which is the SSE standard.
        """

        async for event in app.astream(init_state, config= config, stream_mode = "updates"):
            #1. check for __interrupt__
            if "__interrupt__" in event:
                interrupt_data = event["__interrupt__"][0].value
                yield f"data: {json.dumps({'type': 'interrupt', 'questions': interrupt_data['questions']})}\n\n"
                return
            
            for node_name in event:
                yield f"data: {json.dumps({'type': 'node_complete', 'node': node_name})}\n\n"
            
        final_state = await app.aget_state(config)

        yield f"data: {json.dumps({'type': 'final_result', 'output': final_state.values.get('final_response')})}"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

class ResumeRequest(BaseModel):
    thread_id: str
    answers: dict[str, str]

@api.post("/resume_stream")
async def resume_pipline(requests: ResumeRequest):
    """
    Resumes the graph's execution after a user provides MCQ answers.
    """
    config = {"configurable": {"thread_id": requests.thread_id}}

    async def event_generator():
        try:
            async for event in app.astream(
                Command(resume=requests.answers),
                config=config,
                stream_mode="updates"
            ):
                for node_name in event:
                    yield f"data: {json.dumps({'type': 'node_complete', 'node': node_name})}\n\n"

            final_state = await app.aget_state(config)
            output = final_state.values.get('final_response', 'No output generated')
            yield f"data: {json.dumps({'type': 'final_result', 'output': output})}\n\n"
            
        except Exception as e:
            raise HTTPException(status_code= 500, detail = str(e))
    return StreamingResponse(event_generator(), media_type="text/event-stream")