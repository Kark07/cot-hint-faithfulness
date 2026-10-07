"""
Model loading and generation. See PROTOCOL.md §2.1 and §3 (prompt formatting, decoding).

Imports are lazy so the dry-run path does not require torch/transformers.
"""

from __future__ import annotations

LETTERS = ("A", "B", "C", "D")


class ModelHandle:
    """One load of the pinned model; chat-templated, batched generation."""

    def __init__(self, name: str, revision: str | None = None, precision: str = "bf16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cuda = torch.cuda.is_available()
        # PROTOCOL.md §2.1: fp16 fallback where bf16 is unsupported (e.g. T4).
        if not cuda:
            precision = "fp32"
        elif precision == "bf16" and not torch.cuda.is_bf16_supported():
            precision = "fp16"
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        self.name = name
        self.revision = revision
        self.precision = precision
        self.device_name = torch.cuda.get_device_name(0) if cuda else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, torch_dtype=dtype,
            device_map="auto" if cuda else None,
        )
        self.model.eval()

        self.letter_ids = {}
        for L in LETTERS:
            ids = self.tokenizer(L, add_special_tokens=False).input_ids
            if len(ids) != 1:
                raise RuntimeError(f"letter {L} is not a single token under this tokenizer")
            self.letter_ids[L] = ids[0]

    def chat(self, user_text: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False, add_generation_prompt=True,
        )

    def generate_batches(
        self,
        user_texts: list[str],
        temperature: float,
        seed: int,
        max_new_tokens: int,
        batch_size: int = 16,
        should_stop=None,
    ):
        """Yield one list of completions per batch, in input order.

        should_stop is an optional zero-argument callable polled at every
        decoding step; when it returns True the in-flight generation stops.
        The caller must then discard that batch (the runner does, by checking
        the deadline before persisting).


        The seed is set once per call; batches run in a fixed order, so the
        sampled outputs are determined by (seed, batch_size, input order, hardware).
        Yielding per batch lets the runner persist progress and enforce the
        compute deadline between batches.
        """
        import torch

        torch.manual_seed(seed)
        sample = temperature > 0
        decode = dict(
            do_sample=sample,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if sample:
            decode.update(temperature=temperature, top_p=1.0, top_k=0)
        else:
            decode.update(temperature=None, top_p=None, top_k=None)

        if should_stop is not None:
            from transformers import StoppingCriteria, StoppingCriteriaList

            class _DeadlineStop(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    return bool(should_stop())

            decode["stopping_criteria"] = StoppingCriteriaList([_DeadlineStop()])

        for i in range(0, len(user_texts), batch_size):
            texts = [self.chat(t) for t in user_texts[i:i + batch_size]]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                                 add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, **decode)
            new = gen[:, enc.input_ids.shape[1]:]
            yield self.tokenizer.batch_decode(new, skip_special_tokens=True)

    def answer_letter_logits(self, user_text: str) -> dict[str, float]:
        """Direct-answer probe (PROTOCOL.md §4.3): logits of A-D after 'Answer: ('."""
        import torch

        text = self.chat(user_text) + "Answer: ("
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0, -1].float()
        return {L: float(logits[i].item()) for L, i in self.letter_ids.items()}
