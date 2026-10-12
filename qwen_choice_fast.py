"""A reusable question plan; images and model outputs are recomputed every call."""
import hashlib

from qwen_choice import QwenChoice


class FastQwenChoice(QwenChoice):
    """Cache only text/template metadata; avoid diagnostic image hashes in the hot path."""

    def prepare(self, image, question, options):
        key = (question, tuple(options.items()))
        if getattr(self, "_question_key", None) != key:
            # Run the reference boundary/option validation once per new question.
            inputs, code_ids, labels, audit = super().prepare(image, question, options)
            codes = list(audit["codes"])
            prompt = question + "\n\nChoices:\n" + "\n".join(
                f"{code}: {label} — {options[label]}" for code, label in zip(codes, labels))
            prompt += "\nReturn exactly one uppercase choice letter. No other text."
            rendered = self.processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": prompt}]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if hashlib.sha256(rendered.encode()).hexdigest() != audit["prompt_sha256"]:
                raise RuntimeError("Prepared question differs from the reference prompt")
            self._question_key = key
            self._question_plan = (rendered, code_ids, labels, audit["prompt_sha256"], audit["codes"])
        else:
            rendered, code_ids, labels, _, _ = self._question_plan
            inputs = self.processor(text=[rendered], images=[image], return_tensors="pt")
        _, code_ids, labels, prompt_sha, codes = self._question_plan
        audit = {
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "image_grid_thw": inputs["image_grid_thw"].tolist(),
            "visual_tokens": int(inputs["image_grid_thw"].prod().item())
            // self.processor.image_processor.merge_size ** 2,
            "prompt_sha256": prompt_sha,
            "input_ids_sha256": None, "pixel_values_sha256": None,
            "input_hashes": "Verify separately; not computed inside each production request",
            "codes": dict(codes), "code_token_ids": list(code_ids),
            "question_plan_reused": True,
        }
        return inputs, list(code_ids), list(labels), audit
