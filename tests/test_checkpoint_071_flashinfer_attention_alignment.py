from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ALIGNMENT = ROOT / "docs" / "flashinfer_attention_alignment.md"


class FlashInferAttentionAlignmentDocumentTests(unittest.TestCase):
    def test_alignment_document_uses_official_sources(self):
        text = ALIGNMENT.read_text(encoding="utf-8")
        self.assertIn("https://docs.flashinfer.ai/api/attention.html", text)
        self.assertIn("flashinfer-ai/flashinfer/blob/main/flashinfer/decode.py", text)
        self.assertIn("single_prefill_with_kv_cache_with_jit_module.html", text)

    def test_high_level_plan_run_contract_is_explicit(self):
        text = ALIGNMENT.read_text(encoding="utf-8")
        self.assertIn("`plan(...) -> None`", text)
        self.assertIn("`run()` 不重新选择另一个实现", text)
        self.assertIn("不把 plan", text)
        self.assertIn("`backend=\"auto\"`", text)

    def test_upstream_and_ascend_internal_architectures_are_not_conflated(self):
        text = ALIGNMENT.read_text(encoding="utf-8")
        self.assertIn("并不是一个让第三方 CANN/FA-NPU 包注册进来的通用 provider registry", text)
        self.assertIn("FlashInfer-NPU 特有的内部适配", text)
        self.assertIn("不声称与上游内部类或源码结构一一对应", text)

    def test_advanced_jit_exception_is_documented(self):
        text = ALIGNMENT.read_text(encoding="utf-8")
        self.assertIn("`*_with_jit_module` 低层入口", text)
        self.assertIn("普通用户使用不带该后缀的函数", text)

    def test_readme_indexes_link_to_alignment_design(self):
        for relative in ("README.md", "docs/README.md"):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("flashinfer_attention_alignment.md", text)

    def test_public_backend_names_do_not_expose_provider_selection(self):
        architecture = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
        frontend = (ROOT / "docs" / "attention_frontend_contract.md").read_text(
            encoding="utf-8"
        )
        self.assertNotIn(
            "允许 `ascendc`、`aclnn`、`reference` 和具体 `kernel_id`", architecture
        )
        self.assertIn("不进入模型侧选择接口", architecture)
        self.assertIn("不加入模型侧参数", frontend)


if __name__ == "__main__":
    unittest.main()
