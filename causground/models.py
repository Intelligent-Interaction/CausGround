"""Two optional model backends; structured-state aggregation imports neither."""
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

from .pipeline import strict_loads

ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def response_text(response):
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
        if choice.get("finish_reason") != "stop" or not isinstance(content, str) or not content.strip():
            raise ValueError("Model response is incomplete or is not text")
        return content
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid model response envelope") from exc


def make_model(backend, *, model=None, model_path=None, timeout=45, allow_images=False):
    if backend not in {"dashscope", "vllm"}:
        raise ValueError("backend must be dashscope or vllm")
    if not 0 < timeout <= 60:
        raise ValueError("timeout must be in (0, 60]")
    if backend == "vllm":
        if model_path is None or not Path(model_path).is_dir():
            raise ValueError("vllm requires an existing --model-path directory")
        from .local_vllm import VllmBackend
        local = VllmBackend(model_path, allow_images=allow_images)
        model = model or Path(model_path).name
    else:
        if model_path is not None:
            raise ValueError("--model-path requires --backend vllm")
        if allow_images and model is None:
            raise ValueError("Image input requires an explicit vision-capable --model")
        model = model or "qwen-plus"
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be nonempty")

    def generate(messages):
        body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 3000,
                "enable_thinking": False, "response_format": {"type": "json_object"}}
        if backend == "vllm":
            return response_text(local.generate(body))
        key = os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise ValueError("Set DASHSCOPE_API_KEY for model inference")
        request = urllib.request.Request(ENDPOINT, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
        try:
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout) as response:
                result = strict_loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"DashScope HTTP {exc.code}; no automatic retry") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise RuntimeError("DashScope connection failed; no automatic retry") from None
        return response_text(result)
    return generate
