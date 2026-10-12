"""Frozen Qwen3.5 text model, one closed-set decision."""
from __future__ import annotations

import hashlib
import string
import time

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM


class QwenChoice:
    """Single caller per instance. No answer or language-state cache."""

    def __init__(self, model_path, device="cuda:0", *, local_files_only=False):
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This accelerated runtime requires an NVIDIA CUDA GPU")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=local_files_only)
        self.model = Qwen3_5ForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map={"": str(self.device)},
            local_files_only=local_files_only).eval().requires_grad_(False)
        self.original_language = self.model.model
        self.compiled_graphs = 0
        self.compiled_graph_calls = 0
        self.model.model.register_forward_pre_hook(self._model_called)
        self.model_calls = 0
        from torch._dynamo.backends.registry import lookup_backend
        inductor = lookup_backend("inductor")

        def counted_backend(graph, example_inputs):
            executable = inductor(graph, example_inputs)
            self.compiled_graphs += 1

            def run(*args, **kwargs):
                self.compiled_graph_calls += 1
                return executable(*args, **kwargs)
            return run

        self.compiled_language = torch.compile(
            self.original_language, backend=counted_backend, dynamic=False,
            fullgraph=False)
        self.compiled_language.register_forward_pre_hook(self._model_called)
        self.kernel_audit = self._check_kernels()

    def _model_called(self, module, args):
        self.model_calls += 1

    @staticmethod
    def _uses_fast_kernel(fn) -> bool:
        """True when transformers hub/package wrapper replaced the torch reference."""
        if not callable(fn):
            return False
        # Legacy: layer stored the imported kernel callable directly.
        if getattr(fn, "__module__", "").startswith(("causal_conv1d", "fla")):
            return True
        closure, code = getattr(fn, "__closure__", None), getattr(fn, "__code__", None)
        if not closure or not code or "is_new_implementation" not in code.co_freevars:
            return False
        return bool(closure[code.co_freevars.index("is_new_implementation")].cell_contents)

    @staticmethod
    def _fused_rmsnorm_active(norm, q) -> bool:
        fused_cls = getattr(q, "FusedRMSNormGated", None)
        if fused_cls is not None and isinstance(norm, fused_cls):
            return True
        try:
            from fla.modules import FusedRMSNormGated as FlAFusedRMSNormGated
        except ImportError:
            FlAFusedRMSNormGated = None
        if FlAFusedRMSNormGated is not None and isinstance(norm, FlAFusedRMSNormGated):
            return True
        # Hub kernels replace Qwen3_5RMSNormGated.forward with a non-reference impl.
        pure = getattr(q.Qwen3_5RMSNormGated, "forward", None)
        fwd = type(norm).forward
        return callable(fwd) and fwd is not pure

    def _check_kernels(self):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q
        layers = [m for m in self.model.modules() if isinstance(m, q.Qwen3_5GatedDeltaNet)]
        if not layers:
            raise RuntimeError("No Qwen3_5GatedDeltaNet layers found for kernel audit")

        sample = layers[0]
        # Older transformers attached kernel callables on each layer; 5.18+ calls
        # module-level wrappers (package or hub) from forward().
        if hasattr(sample, "causal_conv1d_fn"):
            delta_fn = getattr(sample, "chunk_gated_delta_rule", None)
            if delta_fn is None:
                delta_fn = getattr(q, "chunk_gated_delta_rule", None)
            status = {
                "causal_conv1d": (
                    sample.causal_conv1d_fn is q.causal_conv1d_fn
                    and callable(q.causal_conv1d_fn)
                    and self._uses_fast_kernel(q.causal_conv1d_fn)),
                "fla_delta": (
                    delta_fn is not None and callable(delta_fn)
                    and self._uses_fast_kernel(delta_fn)),
                "fla_norm": self._fused_rmsnorm_active(sample.norm, q),
            }
        else:
            delta_fn = getattr(q, "torch_chunk_gated_delta_rule", None) or getattr(
                q, "chunk_gated_delta_rule", None)
            status = {
                "causal_conv1d": self._uses_fast_kernel(q.causal_conv1d_fn),
                "fla_delta": self._uses_fast_kernel(delta_fn),
                "fla_norm": self._fused_rmsnorm_active(sample.norm, q),
            }

        if not all(status.values()):
            missing = [name for name, ok in status.items() if not ok]
            raise RuntimeError(
                "Full acceleration requires causal-conv1d and FLA "
                f"(missing/inactive: {', '.join(missing)}); no silent fallback. "
                "Install causal-conv1d and flash-linear-attention, or enable "
                "transformers hub kernels with use_kernels=True.")
        return {
            "linear_attention_layers": len(layers),
            "all_fast_kernels_active": True,
            **status,
        }

    def prepare(self, question, options):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Supply exactly one nonempty question string")
        if (not isinstance(options, dict) or not 2 <= len(options) <= 26
                or any(not isinstance(k, str) or not k.strip()
                       or not isinstance(v, str) or not v.strip()
                       for k, v in options.items())):
            raise ValueError("Options must map 2-26 nonempty names to descriptions")
        codes = list(string.ascii_uppercase[:len(options)])
        labels = list(options)
        prompt = question + "\n\nChoices:\n" + "\n".join(
            f"{code}: {label} — {options[label]}"
            for code, label in zip(codes, labels))
        prompt += "\nReturn exactly one uppercase choice letter. No other text."
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if not rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"):
            raise RuntimeError("Unexpected non-thinking chat-template suffix")
        # Check continuation tokens at the actual assistant boundary, not in isolation.
        fence = rendered[rendered.rfind("<|im_start|>"):]
        prefix = self.tokenizer.encode(fence, add_special_tokens=False)
        code_ids = []
        for code in codes:
            combined = self.tokenizer.encode(fence + code, add_special_tokens=False)
            if combined[:len(prefix)] != prefix or len(combined) != len(prefix) + 1:
                raise ValueError("Choice code is not one stable continuation token")
            code_ids.append(combined[-1])
        inputs = self.tokenizer([rendered], return_tensors="pt")
        if inputs["input_ids"][0, -len(prefix):].tolist() != prefix:
            raise RuntimeError("Processed input changed the answer boundary")
        audit = {
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "input_ids_sha256": hashlib.sha256(inputs["input_ids"].numpy().tobytes()).hexdigest(),
            "codes": dict(zip(codes, labels)), "code_token_ids": code_ids,
        }
        return inputs, code_ids, labels, audit

    def _transfer(self, inputs):
        transferred = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if value.device.type == "cpu" and value.nbytes >= 4096:
                    value = value.pin_memory()
                value = value.to(self.device, non_blocking=True)
            transferred[key] = value
        return transferred

    @torch.inference_mode()
    def judge(self, question, options, *, profile="accelerated"):
        if profile not in {"accelerated", "eager", "native"}:
            raise ValueError("Unknown profile")
        use_compile = profile == "accelerated"
        self.model.model = self.compiled_language if use_compile else self.original_language
        graph_calls_before = self.compiled_graph_calls
        model_calls_before = self.model_calls
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        try:
            cpu_inputs, code_ids, labels, audit = self.prepare(question, options)
            prepared_at = time.perf_counter()
            inputs = self._transfer(cpu_inputs)
            model_start = torch.cuda.Event(enable_timing=True)
            model_end = torch.cuda.Event(enable_timing=True)
            model_start.record()
            with torch._dynamo.config.patch(recompile_limit=64):
                if profile == "native":
                    sequence = self.model.generate(
                        **inputs, do_sample=False, max_new_tokens=8, use_cache=True)
                    model_end.record()
                    tokens = sequence[0, inputs["input_ids"].shape[-1]:].tolist()
                    answer = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    eos = self.model.generation_config.eos_token_id
                    eos = [eos] if isinstance(eos, int) else eos
                    complete = bool(tokens and tokens[-1] in eos)
                    valid = complete and answer in audit["codes"]
                    result = {"choice": audit["codes"].get(answer), "answer_code": answer,
                              "probabilities": None, "generated_tokens": len(tokens),
                              "complete": complete, "valid": valid}
                else:
                    # The original full BF16 head is applied only to the final hidden state.
                    # No autoregressive decoding and no per-option forward passes.
                    hidden = self.model.model(**inputs, use_cache=False,
                                              past_key_values=None).last_hidden_state
                    logits = self.model.lm_head(hidden[:, -1:, :])[0, -1].float()
                    ids = torch.tensor(code_ids, device=self.device)
                    values = logits.index_select(0, ids)
                    probabilities = values.softmax(dim=0)
                    option_mass = (values.logsumexp(0) - logits.logsumexp(0)).exp()
                    model_end.record()
                    scores = probabilities.cpu().tolist()
                    selected = max(range(len(scores)), key=scores.__getitem__)
                    result = {"choice": labels[selected], "answer_code": list(audit["codes"])[selected],
                              "probabilities": dict(zip(labels, scores)),
                              "candidate_logits": values.cpu().tolist(),
                              "unconstrained_option_mass": float(option_mass),
                              "generated_tokens": 0, "complete": True, "valid": True}
            torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - started
        finally:
            self.model.model = self.original_language
        result.update({"profile": profile, "request_s": elapsed,
                       "prepare_s": prepared_at - started,
                       "model_gpu_s": model_start.elapsed_time(model_end) / 1000,
                       "audit": dict(audit,
                                     model_forward_calls=self.model_calls - model_calls_before,
                                     compiled_graph_calls=self.compiled_graph_calls - graph_calls_before,
                                     compiled_graphs_total=self.compiled_graphs,
                                     enable_thinking=False, batch_size=1,
                                     cross_request_cache=False, generate_called=profile == "native")})
        if use_compile and result["audit"]["compiled_graph_calls"] == 0:
            raise RuntimeError("Compilation was requested but no compiled graph ran")
        return result
