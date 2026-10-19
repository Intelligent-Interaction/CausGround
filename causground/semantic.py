"""Independent mechanism ranking; lexical baseline and optional embeddings."""
from collections import Counter
import math
import re


class LexicalRetriever:
    """Small offline baseline, explicitly NOT a semantic embedding model."""
    name = "lexical_tfidf"

    def __call__(self, task, documents):
        tokens = lambda s: re.findall(r"[\w]+", s.lower().replace("_", " "))
        query = Counter(tokens(task))
        docs = [Counter(tokens(d["text"])) for d in documents]
        df = Counter(t for d in docs for t in d)
        def vector(counter):
            return {t: (1 + math.log(n)) * (1 + math.log((1 + len(docs)) / (1 + df[t]))) for t, n in counter.items()}
        q = vector(query)
        def cosine(d):
            v = vector(d)
            scale = math.sqrt(sum(x*x for x in q.values()) * sum(x*x for x in v.values()))
            return sum(q[t]*v.get(t, 0) for t in q) / scale if scale else 0.0
        return [cosine(d) for d in docs]


class EmbeddingRetriever:
    """SentenceTransformer or compatible encoder, loaded only when requested."""
    name = "sentence_transformers"

    def __init__(self, model="Qwen/Qwen3-Embedding-0.6B", *, encoder=None, device=None, query_prefix=None):
        self.name = "sentence_transformers" if encoder is None else getattr(encoder, "name", "injected_encoder")
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(model, device=device)
        self.encoder, self.model = encoder, model
        self.query_prefix = query_prefix if query_prefix is not None else ("Instruct: Given an embodied task, retrieve relevant action mechanisms, conditions and effects.\nQuery: " if "qwen3" in str(model).lower() else "")
        self._key, self._vectors = None, None

    def __call__(self, task, documents):
        key = tuple(d["text"] for d in documents)
        if self._key != key:
            self._vectors = self.encoder.encode(list(key), normalize_embeddings=True, show_progress_bar=False)
            self._key = key
        q = self.encoder.encode([self.query_prefix + task], normalize_embeddings=True, show_progress_bar=False)[0]
        def cosine(v):
            if len(q) != len(v): raise ValueError("Embedding dimensions differ")
            norm = math.sqrt(sum(float(x)**2 for x in q) * sum(float(x)**2 for x in v))
            return sum(float(a)*float(b) for a,b in zip(q,v)) / norm if norm else 0.0
        return [cosine(v) for v in self._vectors]


class Qwen3Encoder:
    """Local-only Qwen3 embedding encoder using the existing Transformers runtime.

    Left padding makes the final hidden token the final non-padding token.
    This backend is specific to Qwen3-Embedding; other encoders use sentence-transformers.
    """
    name = "qwen3_transformers"

    def __init__(self, model_path, *, device="cpu", batch_size=4, max_length=512):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.device = torch, device
        self.batch_size, self.max_length = batch_size, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32).to(device).eval()

    def encode(self, texts, **kwargs):
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[start:start+self.batch_size], padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                values = self.model(**batch).last_hidden_state[:, -1]
                values = self.torch.nn.functional.normalize(values, p=2, dim=1)
            vectors.extend(values.cpu().tolist())
        return vectors
