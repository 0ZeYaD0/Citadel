from pathlib import Path

from citadel.api.prompt_loader import load_study_prompts
from citadel.core.concurrency import DirectFileOperations


class CountingFileOperations(DirectFileOperations):
    def __init__(self) -> None:
        self.read_count = 0

    def read_text(self, path: Path, *, encoding: str = "utf-8") -> str:
        self.read_count += 1
        return super().read_text(path, encoding=encoding)


def _prompt_catalog() -> str:
    return "\n".join(
        [
            "[flashcards]",
            "Create {count} cards.",
            "[quiz]",
            "Create {count} questions.",
            "[summary]",
            "Summarize the passages.",
            "[key_terms]",
            "List {count} terms.",
            "[explain_simple]",
            "Explain simply.",
        ]
    )


def test_prompt_loader_reuses_unchanged_file(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text(_prompt_catalog(), encoding="utf-8")
    operations = CountingFileOperations()

    first = load_study_prompts(path, operations)
    second = load_study_prompts(path, operations)

    assert first == second
    assert operations.read_count == 1


def test_prompt_loader_reloads_modified_file(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text(_prompt_catalog(), encoding="utf-8")
    operations = CountingFileOperations()

    load_study_prompts(path, operations)
    path.write_text(_prompt_catalog().replace("Create {count} cards.", "Make {count} cards."), encoding="utf-8")

    loaded = load_study_prompts(path, operations)

    assert loaded["flashcards"] == "Make {count} cards."
    assert operations.read_count == 2
