import subprocess
from pathlib import Path

repo = Path("data/PlantDoc").resolve()
if not (repo / ".git").exists():
    raise SystemExit("ERROR: Git repository not found.")

listing = subprocess.run(
    ["git", "-C", str(repo), "ls-tree", "-rz", "--full-tree", "HEAD"],
    stdout=subprocess.PIPE,
    check=True,
).stdout

reserved = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
bad_chars = set('<>:"\\|?*')

written = 0
skipped = 0
failed = 0

for record in listing.split(b"\0"):
    if not record:
        continue

    metadata, raw_path = record.split(b"\t", 1)
    mode, kind, oid = metadata.decode("ascii").split()
    relative = raw_path.decode("utf-8", errors="surrogateescape")
    parts = relative.split("/")

    if kind != "blob":
        continue

    # Only extract dataset split files, not repository metadata.
    if parts[0] not in ("train", "test"):
        continue

    invalid = any(
        any(c in bad_chars for c in part)
        or part.endswith((".", " "))
        or part.split(".")[0].upper() in reserved
        for part in parts
    )

    if invalid:
        print("SKIP:", relative)
        skipped += 1
        continue

    destination = repo.joinpath(*parts)

    # Guard against paths escaping the dataset directory.
    if not destination.resolve().is_relative_to(repo):
        print("SKIP unsafe path:", relative)
        skipped += 1
        continue

    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with destination.open("wb") as output:
            subprocess.run(
                ["git", "-C", str(repo), "cat-file", "blob", oid],
                stdout=output,
                check=True,
            )
        written += 1
        if written % 100 == 0:
            print(f"Extracted {written} files...")
    except Exception as exc:
        print("FAILED:", relative, exc)
        failed += 1

print("\nExtraction summary")
print("Files extracted:", written)
print("Paths skipped:", skipped)
print("Files failed:", failed)
