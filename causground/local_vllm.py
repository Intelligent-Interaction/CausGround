"""Optional local generation backend; vLLM is imported only on actual inference."""
from pathlib import Path
import os


class LocalInferenceError(RuntimeError):
    pass


class VllmBackend:
    def __init__(self, model_path, *, allow_images=False):
        self.model_path = str(Path(model_path).resolve())
        self.allow_images = allow_images
        self.engine = None

    def generate(self, body):
        try:
            has_images = any(isinstance(m.get("content"), list) and
                             any(part.get("type") == "image_url" for part in m["content"])
                             for m in body["messages"])
            if has_images and not self.allow_images:
                raise ValueError("Create the local model with allow_images=True for image input")
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import StructuredOutputsParams
            if self.engine is None:
                self.engine = LLM(
                    model=self.model_path, tokenizer=self.model_path, tensor_parallel_size=1,
                    dtype="bfloat16", gpu_memory_utilization=float(os.environ.get("CAUSGROUND_VLLM_GPU_MEMORY_UTILIZATION", "0.70")), max_model_len=8192,
                    max_num_seqs=1, enforce_eager=True, trust_remote_code=False, seed=0,
                    enable_prefix_caching=False, limit_mm_per_prompt={"image": int(self.allow_images), "video": 0},
                    disable_log_stats=True)
            params = SamplingParams(
                temperature=body["temperature"], max_tokens=body["max_tokens"], seed=0,
                structured_outputs=StructuredOutputsParams(json_object=True))
            output = self.engine.chat(body["messages"], sampling_params=params, use_tqdm=False,
                                      chat_template_kwargs={"enable_thinking": False})[0]
            completion = output.outputs[0]
            return {"choices": [{"finish_reason": completion.finish_reason,
                                 "message": {"content": completion.text}}]}
        except Exception as exc:
            raise LocalInferenceError(type(exc).__name__ + ": " + str(exc)) from exc
