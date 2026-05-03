import unittest


class LayoutSmokeTest(unittest.TestCase):
    def test_package_imports(self) -> None:
        from drivesense.backend import chatbot, speech, vision
        from drivesense.benchmarks import llm_benchmark, summarize_timm_benchmark
        from drivesense.data import prepare_dataset, repair_affectnet_labels
        from drivesense.frontend import gui
        from drivesense.training import train_emotion_timm, train_eye_timm

        self.assertTrue(chatbot.DEFAULT_MODEL)
        self.assertTrue(hasattr(speech, "main"))
        self.assertTrue(hasattr(vision, "main"))
        self.assertTrue(hasattr(gui, "DriverAssistantWindow"))
        self.assertTrue(hasattr(prepare_dataset, "main"))
        self.assertTrue(hasattr(train_emotion_timm, "main"))
        self.assertTrue(hasattr(train_eye_timm, "main"))
        self.assertTrue(hasattr(llm_benchmark, "SCENARIOS"))
        self.assertTrue(hasattr(repair_affectnet_labels, "main"))
        self.assertTrue(hasattr(summarize_timm_benchmark, "main"))


if __name__ == "__main__":
    unittest.main()
