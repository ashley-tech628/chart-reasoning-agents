"""Streaming access to the DVQA tar.gz archives.

The official DVQA release ships three gzip-compressed tarballs:

    data/dvqa/
        qa.tar.gz         # the three {split}_qa.json files (~750 MB uncompressed)
        images.tar.gz     # ~300K bar-chart PNGs (~6.5 GB uncompressed)
        metadata.tar.gz   # per-bar bbox + value annotations

A full extraction wastes ~10 GB of disk just to read a 500-question subsample.
This module gives two helpers that stream the tars instead:

- `read_qa_split_from_tar(qa_tar, [splits])` parses just the JSON file(s) we
  need straight into memory, never writing them to disk.

- `extract_images_from_tar(images_tar, needed_filenames, out_dir)` streams
  the image tarball once and saves only the files whose basenames are in
  `needed_filenames`. It breaks early as soon as the entire needed set has
  been seen, so for a few hundred images you only pay for the prefix of
  the archive where those images live.

Both helpers open the tar with `mode='r|gz'` (the streaming mode); they make
exactly one forward pass and never seek, so they work even if the tar is on
a slow remote filesystem.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _basename(name: str) -> str:
    """Posix basename — tar members always use forward slashes."""
    return name.rsplit("/", 1)[-1]


def iter_tar_member_names(tar_path: str | Path) -> Iterator[str]:
    """One-pass streaming listing of every member name in the tar.

    Useful for a sanity check ("does this tar actually contain the splits I
    expect?") without paying to materialize anything. Does NOT load file
    contents — only the tar header for each member is touched.
    """
    with tarfile.open(str(tar_path), mode="r|gz") as tf:
        for member in tf:
            yield member.name


def peek_tar_summary(tar_path: str | Path, max_members: int = 5) -> Dict:
    """Quick listing for diagnostics. Streams just far enough to count entries
    and surface the first few names — handy before kicking off a long run."""
    sample: List[str] = []
    total = 0
    total_bytes = 0
    with tarfile.open(str(tar_path), mode="r|gz") as tf:
        for member in tf:
            total += 1
            if member.isfile():
                total_bytes += member.size
            if len(sample) < max_members:
                sample.append(member.name)
    return {
        "path": str(tar_path),
        "n_members": total,
        "first_members": sample,
        "uncompressed_bytes": total_bytes,
    }


# ---------------------------------------------------------------------------
# qa.tar.gz: pull {split}_qa.json straight into memory
# ---------------------------------------------------------------------------


def _matches_qa_split(member_name: str, splits: Set[str]) -> Optional[str]:
    """If a tar member is the {split}_qa.json file for one of the requested
    splits, return that split name, else None."""
    base = _basename(member_name)
    for split in splits:
        if base == f"{split}_qa.json":
            return split
    return None


def read_qa_split_from_tar(
    qa_tar_path: str | Path,
    splits: Iterable[str],
) -> Dict[str, list]:
    """Stream `qa.tar.gz` and return `{split: [qa_record, ...], ...}` for the
    requested splits, parsing each JSON file directly from the tar stream.

    One streaming pass extracts every requested split in a single read. We
    break out of the loop as soon as we've collected all of them.
    """
    splits = set(splits)
    out: Dict[str, list] = {}
    with tarfile.open(str(qa_tar_path), mode="r|gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            which = _matches_qa_split(member.name, splits)
            if which is None or which in out:
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            # json.load reads incrementally from the streamed file object.
            out[which] = json.load(f)
            if set(out.keys()) == splits:
                break

    missing = splits - set(out.keys())
    if missing:
        raise FileNotFoundError(
            f"qa.tar.gz did not contain JSON files for splits: {sorted(missing)}. "
            f"Found members like: {sample_member_names(qa_tar_path, 6)}"
        )
    return out


def sample_member_names(tar_path: str | Path, k: int = 6) -> List[str]:
    """Return up to `k` member names from the tar (best-effort, for error msgs)."""
    names: List[str] = []
    try:
        for name in iter_tar_member_names(tar_path):
            names.append(name)
            if len(names) >= k:
                break
    except Exception:  # pragma: no cover
        pass
    return names


# ---------------------------------------------------------------------------
# images.tar.gz: stream-extract only the basenames we need
# ---------------------------------------------------------------------------


def extract_images_from_tar(
    images_tar_path: str | Path,
    needed_filenames: Iterable[str],
    out_dir: str | Path,
    *,
    skip_existing: bool = True,
    progress_every: int = 100,
    log: Optional[callable] = None,
) -> Dict:
    """Stream `images.tar.gz` once and copy out only the images whose basename
    is in `needed_filenames`. Stops as soon as the whole set has been found.

    Files are written flat under `out_dir/` (any tarball-internal directory
    structure is collapsed), so the downstream runner can use
    `out_dir / image_filename` as the absolute path.

    Returns a dict summary with counts and any missing names.
    """
    needed: Set[str] = set(needed_filenames)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if log is None:
        log = lambda msg: print(msg, flush=True)  # noqa: E731

    if skip_existing:
        already_have = {n for n in needed if (out_dir / n).exists()}
    else:
        already_have = set()
    to_get: Set[str] = needed - already_have

    log(
        f"Extraction plan: {len(needed)} needed, "
        f"{len(already_have)} already on disk, {len(to_get)} to extract."
    )

    if not to_get:
        return {
            "needed": len(needed),
            "already_present": len(already_have),
            "extracted": 0,
            "missing": [],
        }

    found: Set[str] = set()
    scanned = 0
    t_start = __import__("time").time()

    with tarfile.open(str(images_tar_path), mode="r|gz") as tf:
        for member in tf:
            scanned += 1
            if not member.isfile():
                continue
            base = _basename(member.name)
            if base not in to_get:
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            target = out_dir / base
            tmp = target.with_suffix(target.suffix + ".part")
            try:
                with open(tmp, "wb") as out:
                    shutil.copyfileobj(f, out, length=1024 * 1024)
                os.replace(tmp, target)
            finally:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:  # pragma: no cover
                        pass
            found.add(base)
            if progress_every and len(found) % progress_every == 0:
                elapsed = __import__("time").time() - t_start
                log(
                    f"  extracted {len(found)}/{len(to_get)} "
                    f"({scanned} members scanned, {elapsed:.1f}s)"
                )
            if found == to_get:
                # We have everything; stop streaming the rest of the tar.
                break

    missing = sorted(to_get - found)
    elapsed = __import__("time").time() - t_start
    log(
        f"Done: extracted {len(found)}/{len(to_get)} in {elapsed:.1f}s "
        f"({scanned} members scanned, {len(missing)} missing)."
    )
    return {
        "needed": len(needed),
        "already_present": len(already_have),
        "extracted": len(found),
        "missing": missing,
        "members_scanned": scanned,
        "elapsed_sec": elapsed,
    }


# ---------------------------------------------------------------------------
# Optional: pull a specific metadata file from metadata.tar.gz
# ---------------------------------------------------------------------------


def read_metadata_split_from_tar(
    metadata_tar_path: str | Path,
    splits: Iterable[str],
) -> Dict[str, list]:
    """Same pattern as `read_qa_split_from_tar` but for `{split}_metadata.json`
    inside metadata.tar.gz. Used for evidence-IoU evaluation, which needs the
    per-bar bboxes."""
    splits = set(splits)
    out: Dict[str, list] = {}
    with tarfile.open(str(metadata_tar_path), mode="r|gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            base = _basename(member.name)
            for split in splits:
                if base == f"{split}_metadata.json" and split not in out:
                    f = tf.extractfile(member)
                    if f is not None:
                        out[split] = json.load(f)
                    break
            if set(out.keys()) == splits:
                break
    return out
