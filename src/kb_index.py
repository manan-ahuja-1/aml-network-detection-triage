"""Day 6 — index the curated AML knowledge base into ChromaDB.

WHAT THE KNOWLEDGE BASE IS FOR
------------------------------
The triage agent has to say WHY an account looks like laundering, in language a
compliance function recognises. It could invent that language, and it would sound
plausible — which is the problem. Retrieval grounds each judgement in text that a
supervisor actually published, so a case note cites FFIEC Appendix F or FinCEN
FIN-2014-A005 rather than the model's own vocabulary.

Every document in kb/corpus/ names a `source_id` registered in kb/sources.json with a
URL and a retrieval date. Verbatim regulatory text is marked as such; our own synthesis
(the typology-to-feature mapping) is marked as ours. A RAG system that cites
unverifiable text is worse than one with no citations at all, because it manufactures
confidence the reader cannot check.

WHY CHUNKING IS NOT OPTIONAL HERE
---------------------------------
ChromaDB's default embedding function is ONNX all-MiniLM-L6-v2, whose max sequence
length is **256 word-piece tokens** — roughly 1,000 characters of English. The corpus
documents run 1,800-2,400 characters, so embedding a whole file would silently discard
more than half of it: the stored text would be complete, the vector would represent only
the opening, and retrieval would quietly fail to match anything in the second half.
Nothing errors. This is the RAG equivalent of the NaN-producing join that cost us the
entity features on Day 2, and it is caught the same way — by checking a number against
expectation rather than trusting the pipeline.

So documents are split at markdown structure boundaries, capped below the model's window,
with the document title prepended to every chunk. The title matters: an isolated bullet
list of red flags embeds poorly without the subject it belongs to.

THE EMBEDDER RUNS LOCALLY
-------------------------
all-MiniLM-L6-v2 executes through onnxruntime on this machine. No API key, no per-query
cost, and the index is reproducible offline. The Anthropic key is needed for the agent,
not for retrieval.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# Below the 256-token window with room for the prepended title. Measured rather than
# guessed: at ~4 characters per token, 256 tokens is ~1,000 characters, and 700 leaves
# headroom for the tokeniser splitting regulatory vocabulary into more pieces than
# ordinary prose ("recordkeeping", "counterparty", "1010.100(xx)").
MAX_CHUNK_CHARS = 700

# ChromaDB's ONNXMiniLM_L6_V2 calls `tokenizer.enable_truncation(max_length=256)`,
# which OVERRIDES the 128 that ships inside the model's own tokenizer.json. Chroma's
# source even comments on the discrepancy. Measuring against the file's 128 would
# report phantom truncation; measuring against 256 is what actually happens at embed
# time. Verified empirically: longest chunk is 194 tokens, so nothing truncates.
EMBEDDER_MAX_TOKENS = 256


def _tokenizer():
    """The embedder's real tokenizer, or None if the model has not been fetched yet."""
    try:
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
        from tokenizers import Tokenizer
    except ImportError:
        return None
    path = (Path(ONNXMiniLM_L6_V2.DOWNLOAD_PATH)
            / ONNXMiniLM_L6_V2.EXTRACTED_FOLDER_NAME / "tokenizer.json")
    if not path.exists():
        return None
    tokenizer = Tokenizer.from_file(str(path))
    # Measure TRUE length: with truncation left on, every long chunk reports exactly
    # the limit and an over-length chunk becomes invisible to this check.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def token_counts(texts: list[str]) -> list[int]:
    """Token length of each chunk under the embedder's own tokenizer. [] if unavailable."""
    tokenizer = _tokenizer()
    if tokenizer is None:
        return []
    return [len(tokenizer.encode(t).ids) for t in texts]


FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def parse_document(path: Path) -> tuple[dict, str]:
    """Split a corpus file into its frontmatter metadata and its body."""
    text = path.read_text()
    match = FRONTMATTER.match(text)
    if not match:
        raise ValueError(f"{path.name} has no frontmatter block; every corpus document "
                         "must declare source_id, title and chunk_type.")

    meta: dict = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip().strip('"')
        if value.startswith("[") and value.endswith("]"):
            # Chroma metadata values must be scalars, so a list becomes a delimited
            # string that a `where` filter can still match with $contains.
            items = [v.strip() for v in value[1:-1].split(",") if v.strip()]
            value = "|".join(items)
        meta[key.strip()] = value

    for required in ("source_id", "title", "chunk_type"):
        if required not in meta:
            raise ValueError(f"{path.name} frontmatter is missing {required!r}")

    return meta, text[match.end():].strip()


def chunk_document(body: str, title: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split a document into embeddable pieces at structural boundaries.

    Paragraphs and bullet blocks are kept whole where they fit, because splitting a
    red-flag list mid-bullet produces a chunk that reads as a fragment and embeds as
    one. A paragraph that alone exceeds the cap is split on sentence boundaries as a
    last resort.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", body) if b.strip()]

    chunks: list[str] = []
    current = ""
    for block in blocks:
        if len(block) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            # Oversized block: split on sentence ends, then hard-wrap anything left.
            sentences = re.split(r"(?<=[.:])\s+", block)
            piece = ""
            for sentence in sentences:
                if len(piece) + len(sentence) + 1 > max_chars and piece:
                    chunks.append(piece.strip())
                    piece = ""
                piece += sentence + " "
            if piece.strip():
                chunks.append(piece.strip())
            continue

        if len(current) + len(block) + 2 > max_chars and current:
            chunks.append(current)
            current = block
        else:
            current = f"{current}\n\n{block}" if current else block

    if current:
        chunks.append(current)

    # The title travels with every chunk. A bare list of funds-transfer red flags and a
    # bare list of shell-company red flags embed almost identically without it.
    return [f"{title}\n\n{c}" for c in chunks]


def load_corpus() -> tuple[list[str], list[dict], list[str]]:
    """Read every corpus document and return (texts, metadatas, ids)."""
    sources = json.loads(config.KB_SOURCES.read_text())
    known = {s["id"] for s in sources["sources"]}

    paths = sorted(config.KB_CORPUS.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no corpus documents in {config.KB_CORPUS}")

    texts: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for path in paths:
        meta, body = parse_document(path)
        if meta["source_id"] not in known:
            raise ValueError(
                f"{path.name} declares source_id {meta['source_id']!r}, which is not "
                "registered in kb/sources.json. Nothing enters the index without "
                "provenance."
            )
        for i, chunk in enumerate(chunk_document(body, meta["title"])):
            texts.append(chunk)
            metadatas.append({
                "source_id": meta["source_id"],
                "title": meta["title"],
                "url": meta.get("url", ""),
                "chunk_type": meta["chunk_type"],
                "typologies": meta.get("typologies", ""),
                "document": path.name,
                "chunk_index": i,
            })
            ids.append(f"{path.stem}:{i}")

    return texts, metadatas, ids


def get_collection(create: bool = False):
    """Open (or create) the persistent Chroma collection."""
    import chromadb

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if create:
        return client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            # Cosine, not the L2 default. These are normalised sentence embeddings, and
            # cosine is what all-MiniLM-L6-v2 was trained against — L2 on unnormalised
            # vectors would let chunk length influence ranking.
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(name=config.CHROMA_COLLECTION)


def build_index(rebuild: bool = False) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if rebuild:
        try:
            client.delete_collection(config.CHROMA_COLLECTION)
            print("  dropped existing collection")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})

    texts, metadatas, ids = load_corpus()
    collection.upsert(documents=texts, metadatas=metadatas, ids=ids)

    print(f"  indexed {len(texts)} chunks from "
          f"{len({m['document'] for m in metadatas})} documents")

    tokens = token_counts(texts)
    if tokens:
        over = [n for n in tokens if n > EMBEDDER_MAX_TOKENS]
        print(f"  tokens per chunk: median {sorted(tokens)[len(tokens) // 2]}, "
              f"max {max(tokens)} of {EMBEDDER_MAX_TOKENS} "
              f"({EMBEDDER_MAX_TOKENS - max(tokens)} spare)")
        if over:
            print(f"  WARNING: {len(over)} chunks exceed the embedder window and will "
                  "be SILENTLY TRUNCATED — their tails are unsearchable")
    print(f"  collection '{config.CHROMA_COLLECTION}' at {config.CHROMA_DIR}")


def retrieve(query: str, k: int | None = None, typology: str | None = None) -> list[dict]:
    """Top-k passages for a query, each carrying its citation.

    `typology` filters to chunks tagged with that pattern type, which is how the agent
    asks for "what does the literature say about fan-in" rather than searching the whole
    corpus and hoping the shape words match.
    """
    k = config.RAG_TOP_K if k is None else k
    collection = get_collection()

    where = {"typologies": {"$contains": typology}} if typology else None
    result = collection.query(query_texts=[query], n_results=k,
                              where_document=None, where=where)

    out = []
    for text, meta, distance in zip(result["documents"][0], result["metadatas"][0],
                                    result["distances"][0]):
        out.append({
            "text": text,
            "title": meta["title"],
            "source_id": meta["source_id"],
            "url": meta["url"],
            "chunk_type": meta["chunk_type"],
            # Cosine distance in [0, 2]; similarity is the more readable direction.
            "similarity": round(1.0 - distance, 4),
        })
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="drop and reindex")
    parser.add_argument("--query", help="run a retrieval and print the hits")
    parser.add_argument("--typology", help="restrict retrieval to one typology tag")
    parser.add_argument("-k", type=int, default=config.RAG_TOP_K)
    args = parser.parse_args()

    if args.query:
        for i, hit in enumerate(retrieve(args.query, args.k, args.typology), 1):
            print(f"\n[{i}] sim {hit['similarity']:.3f}  {hit['title']}  "
                  f"({hit['source_id']})")
            print("    " + hit["text"].replace("\n", "\n    ")[:600])
    else:
        build_index(rebuild=args.rebuild)
