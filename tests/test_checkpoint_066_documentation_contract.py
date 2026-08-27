from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


class AttentionDocumentationContractTests(unittest.TestCase):
    def test_provider_onboarding_is_linked_from_entry_documents(self):
        for path in (ROOT / "README.md", DOCS / "README.md"):
            with self.subTest(path=path.name):
                self.assertIn(
                    "attention_provider_onboarding.md",
                    path.read_text(encoding="utf-8"),
                )

    def test_provider_onboarding_freezes_public_and_bootstrap_boundaries(self):
        text = (DOCS / "attention_provider_onboarding.md").read_text(
            encoding="utf-8"
        )
        required = (
            "AttentionOperatorPackageRuntimeSpec",
            "build_attention_operator_runtime_resolvers",
            "install_attention_operator_runtime_resolvers",
            "validate_provider_results",
            "AttentionOperatorQuantizedKVInput",
            "AttentionOperatorQuantizedTensorInput",
            "last_run_receipt",
            "默认 registry 仍为空",
            "不触发另一个 provider 的隐式 retry",
            '不得为了接入方便而增加\n`provider="cann"`',
        )
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_stale_framework_gap_claims_are_absent(self):
        corpus = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "README.md", *sorted(DOCS.glob("*.md")))
        )
        stale_claims = (
            "AttentionOperatorQuantizedKVInput` for a future provider integration",
            "真实 Torch\n   tensor metadata 与 package adapter 尚未接入",
        )
        for claim in stale_claims:
            with self.subTest(claim=claim):
                self.assertNotIn(claim, corpus)

    def test_repository_docs_do_not_embed_run_results(self):
        forbidden = re.compile(
            r"(?:Ran\s+\d+\s+tests|\d+\s+tests?\s+(?:passed|通过)|"
            r"checkpoint\s*\d+\s*(?:passed|通过)|测试结果|验证结果)",
            re.IGNORECASE,
        )
        for path in (ROOT / "README.md", *sorted(DOCS.glob("*.md"))):
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertIsNone(forbidden.search(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
