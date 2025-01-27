import argparse
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.entrypoints.api_models import GenerationRequest
from vllm.utils import random_uuid
from vllm.entrypoints.exceptions import ServerException, server_exception_contextmanager

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
engine = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.exception_handler(ServerException)
async def exception_handler(request: Request, exc: ServerException):
    return JSONResponse(
        status_code=400,
        content={"message": f"App error: Reason: {exc.reason}"},
    )


@app.post("/generate")
async def generate(request: GenerationRequest) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - sampling_params: the sampling parameters .
    """
    request_id = random_uuid()

    with server_exception_contextmanager():
        results_generator = engine.generate(
            request.prompt, SamplingParams(**dict(request.sampling_params)), request_id
        )

    # Streaming case
    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)


@app.post("/generate_streaming")
async def generate_streaming(request: GenerationRequest) -> Response:
    request_id = random_uuid()
    results_generator = engine.generate(
        request.prompt, request.sampling_params, request_id
    )

    results_generator = engine.generate(
        request.prompt, SamplingParams(**dict(request.sampling_params)), request_id
    )

    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [prompt + output.text for output in request_output.outputs]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    return StreamingResponse(stream_results())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
