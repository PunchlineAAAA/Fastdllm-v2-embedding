import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch


class OfficialMTEBWrapper:
    def __init__(self, model, task_to_instructions: dict[str, str] | None = None):
        self.model = model
        self.task_to_instructions = task_to_instructions or {}

    def encode(
        self,
        sentences: list[str],
        *,
        prompt_name: str | None = None,
        batch_size: int = 32,
        **kwargs,
    ):
        instruction = self.task_to_instructions.get(prompt_name, "") if prompt_name else ""
        pairs = [[instruction, sentence] for sentence in sentences]
        return self.model.encode(pairs, batch_size=batch_size, **kwargs)

    def encode_queries(self, queries: list[str], **kwargs):
        return self.encode(queries, **kwargs)

    def encode_corpus(self, corpus, **kwargs):
        if isinstance(corpus, dict):
            titles = corpus.get("title", [""] * len(corpus["text"]))
            texts = corpus["text"]
            pairs = [["", " ".join(part for part in [title, text] if part).strip()] for title, text in zip(titles, texts)]
        else:
            pairs = []
            for doc in corpus:
                if isinstance(doc, dict):
                    pairs.append(["", " ".join(part for part in [doc.get("title", ""), doc.get("text", "")] if part).strip()])
                else:
                    pairs.append(["", str(doc)])
        batch_size = kwargs.pop("batch_size", 32)
        kwargs.pop("request_qid", None)
        kwargs.pop("prompt_name", None)
        return self.model.encode(pairs, batch_size=batch_size, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an LLM2Vec checkpoint with official MTEB v2.")
    parser.add_argument("--base_model_name_or_path", required=True)
    parser.add_argument("--peft_model_name_or_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--benchmark_name", default="MTEB(eng, v2)")
    parser.add_argument(
        "--task_names",
        default=None,
        help="Optional comma-separated task names. If set, overrides --benchmark_name.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--task_to_instructions_fp", default=None)
    parser.add_argument("--bidirectional", action="store_true", default=True)
    parser.add_argument("--no_bidirectional", dest="bidirectional", action="store_false")
    parser.add_argument(
        "--torch_dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    return parser.parse_args()


def load_official_mteb():
    try:
        import mteb
    except ImportError as exc:
        raise ImportError(
            "Official MTEB v2 is not installed. Please install `mteb>=2.0` in the active environment."
        ) from exc

    try:
        installed_version = version("mteb")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Could not read the installed MTEB package metadata. Please reinstall `mteb>=2.0`."
        ) from exc

    major_version = int(installed_version.split(".", 1)[0])
    if major_version < 2:
        raise RuntimeError(
            f"Installed MTEB version is {installed_version}, but this script requires official MTEB v2 or newer."
        )
    return mteb, installed_version


def main():
    args = parse_args()
    mteb, installed_version = load_official_mteb()
    from llm2vec import LLM2Vec

    task_to_instructions = None
    if args.task_to_instructions_fp is not None:
        with open(args.task_to_instructions_fp, "r", encoding="utf-8") as fin:
            task_to_instructions = json.load(fin)

    torch_dtype = (
        args.torch_dtype if args.torch_dtype == "auto" else getattr(torch, args.torch_dtype)
    )
    l2v_model = LLM2Vec.from_pretrained(
        args.base_model_name_or_path,
        peft_model_name_or_path=args.peft_model_name_or_path,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        enable_bidirectional=args.bidirectional,
        torch_dtype=torch_dtype,
        merge_peft=True,
    )

    model = OfficialMTEBWrapper(model=l2v_model, task_to_instructions=task_to_instructions)

    if args.task_names:
        task_names = [task.strip() for task in args.task_names.split(",") if task.strip()]
        tasks = mteb.get_tasks(tasks=task_names)
    else:
        tasks = mteb.get_benchmark(args.benchmark_name)

    print(f"Using official MTEB version: {installed_version}")
    print(f"Evaluating tasks from: {args.task_names or args.benchmark_name}")

    mteb.evaluate(
        model,
        tasks=tasks,
        output_folder=args.output_dir,
        encode_kwargs={"batch_size": args.batch_size},
    )


if __name__ == "__main__":
    main()
