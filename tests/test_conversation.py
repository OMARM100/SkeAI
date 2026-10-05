import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.skeai.level2.chat import ConversationMemory, SkeAIConversation


class ConversationMemoryTests(unittest.TestCase):
    def test_memory_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            memory = ConversationMemory()
            memory.remember_from_user("اسمي عمر")
            memory.add_turn("اسمي عمر", "تشرفت بك يا عمر.")
            memory.save(path)

            loaded = ConversationMemory.load(path)

        self.assertEqual(loaded.facts["user_name"], "عمر")
        self.assertEqual(loaded.state.user_facts["user_name"], "عمر")
        self.assertEqual(loaded.turns[-1]["assistant"], "تشرفت بك يا عمر.")

    def test_chat_keeps_context_and_persists(self) -> None:
        class FakeTokenizer:
            def encode(self, text, **kwargs):
                return list(range(len(text)))

            def decode(self, token_ids, **kwargs):
                return "decoded"

        class FakeConfig:
            context_length = 128

        class FakeModel:
            config = FakeConfig()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            conversation = SkeAIConversation(
                FakeModel(),
                FakeTokenizer(),
                memory_path=path,
            )

            prompts = []

            def fake_generate(model, tokenizer, prompt, **kwargs):
                prompts.append(prompt)
                return "رد تجريبي"

            with patch("src.skeai.level2.chat.generate_text", fake_generate):
                first = conversation.chat("اسمي عمر")
                second = conversation.chat("هل تتذكر اسمي؟")

            saved = ConversationMemory.load(path)

        self.assertEqual(first, "رد تجريبي")
        self.assertEqual(second, "رد تجريبي")
        self.assertIn("حالة SkeAI:", prompts[-1])
        self.assertIn("عمر", prompts[-1])
        self.assertIn("أنا لست إنسانًا.", prompts[-1])
        self.assertIn("المستخدم: اسمي عمر", prompts[-1])
        self.assertIn("SkeAI: رد تجريبي", prompts[-1])
        self.assertEqual(saved.facts["user_name"], "عمر")
        self.assertEqual(len(saved.turns), 2)
        self.assertEqual(saved.state.self_model["name"], "SkeAI")
        self.assertEqual(saved.state.user_facts["user_name"], "عمر")


if __name__ == "__main__":
    unittest.main()


    def test_user_name_can_be_marked_as_system_username(self) -> None:
        memory = ConversationMemory()
        memory.remember_from_user("اسمي عمر")
        memory.remember_from_user("عمر مجرد اسم مستخدم")
        self.assertEqual(memory.user_facts["user_name"], "عمر")
        self.assertEqual(memory.user_facts["username_is_real_name"], "false")

    def test_developmental_experience_is_recorded(self) -> None:
        memory = ConversationMemory()
        memory.add_turn("مرحبا", "أهلًا بك.")
        self.assertTrue(memory.state.experiences)
        self.assertEqual(memory.state.self_model["entity_type"], "ذكاء اصطناعي")
