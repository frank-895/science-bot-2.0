"""Sequence and alignment summary tools for resolution."""

import zipfile
from pathlib import Path
from typing import Literal

from science_bot.pipeline.resolution.tools.reader import FileRef, parse_filename
from science_bot.pipeline.resolution.tools.schemas import FastaSummary

DNA_ALPHABET = set("ACGTUN-")
PROTEIN_ALPHABET = set("ABCDEFGHIKLMNPQRSTVWXYZ*-")


def summarize_fasta_file(capsule_path: Path, filename: str) -> FastaSummary:
    """Summarize a FASTA-like sequence or alignment file.

    Args:
        capsule_path: Absolute path to the capsule directory.
        filename: Direct filename or zip-contained FASTA-like path.

    Returns:
        FastaSummary: Deterministic summary statistics for the file.

    Raises:
        ValueError: If the file is not FASTA-like or contains no sequences.
    """
    ref = parse_filename(capsule_path, filename)
    sequences = _read_fasta_sequences(ref)
    if not sequences:
        raise ValueError(f"File '{filename}' does not contain FASTA records.")

    lengths = [len(sequence) for sequence in sequences]
    total_characters = sum(lengths)
    gap_count = sum(sequence.count("-") for sequence in sequences)
    gap_fraction = float(gap_count / total_characters) if total_characters > 0 else None

    return FastaSummary(
        filename=filename,
        sequence_count=len(sequences),
        min_length=min(lengths) if lengths else None,
        max_length=max(lengths) if lengths else None,
        mean_length=(float(total_characters) / len(sequences)) if sequences else None,
        total_characters=total_characters,
        gap_count=gap_count,
        gap_fraction=gap_fraction,
        alphabet_hint=_infer_alphabet_hint(sequences),
    )


def _read_fasta_sequences(ref: FileRef) -> list[str]:
    sequences: list[str] = []
    current: list[str] = []
    saw_header = False

    for line in _iter_text_lines(ref):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            saw_header = True
            if current:
                sequences.append("".join(current))
                current = []
            continue
        if not saw_header:
            return []
        current.append(stripped)

    if current:
        sequences.append("".join(current))
    return sequences


def _iter_text_lines(ref: FileRef) -> list[str]:
    if ref.zip_path is not None and ref.inner_path is not None:
        with zipfile.ZipFile(ref.zip_path) as archive:
            with archive.open(ref.inner_path) as handle:
                return [line.decode("utf-8", errors="replace") for line in handle]
    with open(ref.file_path, encoding="utf-8", errors="replace") as handle:
        return list(handle)


def _infer_alphabet_hint(
    sequences: list[str],
) -> Literal["dna", "protein", "unknown"]:
    characters = {character.upper() for sequence in sequences for character in sequence}
    characters.discard(" ")
    if not characters:
        return "unknown"
    if characters <= DNA_ALPHABET:
        return "dna"
    if characters <= PROTEIN_ALPHABET:
        return "protein"

    non_gap = {character for character in characters if character != "-"}
    if non_gap and len(non_gap & DNA_ALPHABET) / len(non_gap) >= 0.8:
        return "dna"
    if non_gap and len(non_gap & PROTEIN_ALPHABET) / len(non_gap) >= 0.8:
        return "protein"
    return "unknown"
