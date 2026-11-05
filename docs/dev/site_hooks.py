"""Expose the canonical examples/ notebooks without duplicating their outputs."""

from pathlib import Path

from mkdocs.plugins import event_priority
from mkdocs.structure.files import File


@event_priority(100)
def on_files(files, config):
    root = Path(config.config_file_path).resolve().parent
    for notebook in sorted((root / "examples").glob("*.ipynb")):
        files.append(File(
            notebook.relative_to(root).as_posix(),
            src_dir=str(root),
            dest_dir=config.site_dir,
            use_directory_urls=config.use_directory_urls,
        ))
    return files
