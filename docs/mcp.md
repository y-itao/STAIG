# STAIG MCP server

The optional local server exposes `staig_presets`, `staig_run` and
`staig_extract_image_features` over stdio using the
[MCP Python SDK's FastMCP API](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x).
It reads local paths and creates local output files. It does not download
datasets; image extraction may download ImageNet weights on first use.

## Install and start

From a STAIG checkout, in your chosen Python environment:

```bash
python -m pip install -e ".[image,cluster,mcp]"
python -m staig.mcp_server
```

The last command waits for an MCP client on stdin; ordinary log output goes to
stderr. MCP is optional and is not imported by the base `staig` package.
FastMCP requires the SDK's 1.x line: an unbounded `mcp>=1.2` currently installs
2.x, where this API was renamed. The
[upstream 1.x documentation](https://github.com/modelcontextprotocol/python-sdk/blob/v1.x/README.md)
recommends a `<2` bound. This server was smoke-tested with `mcp==1.30.0`.

The package registers the following optional dependency and entry point in
`pyproject.toml`, so `staig-mcp` also starts the server after installation:

```toml
[project.scripts]
staig-mcp = "staig.mcp_server:main"

[project.optional-dependencies]
mcp = ["mcp>=1.2,<2"]
```

Keep the `<2` bound when maintaining this FastMCP implementation. For older
checkouts without these packaging entries, install `mcp>=1.2,<2` separately and
use `python -m staig.mcp_server`.

## Claude Desktop and Claude Code

Merge this entry into `claude_desktop_config.json` (Desktop: Settings → Developer
→ Edit Config) or the project's `.mcp.json` for Claude Code. Replace the Python
path with the **absolute interpreter path** of the environment where STAIG and
MCP were installed; the Windows equivalent ends in `python.exe`. Do not replace
other servers in an existing file.

```json
{
  "mcpServers": {
    "staig": {
      "command": "/absolute/path/to/environment/bin/python",
      "args": ["-m", "staig.mcp_server"],
      "env": {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "4"
      }
    }
  }
}
```

This configuration selects CPU operation. CUDA use requires a suitable torch
build and removing the empty `CUDA_VISIBLE_DEVICES`. You can also register the
stdio command with Claude Code:

```bash
claude mcp add --transport stdio --scope project staig -- /absolute/path/to/environment/bin/python -m staig.mcp_server
```

Configuration references: [Claude Code MCP](https://code.claude.com/docs/en/mcp)
and [connecting local servers to Claude Desktop](https://modelcontextprotocol.io/docs/develop/connect-local-servers).
Registration is shown as documentation; this task did not modify a user's client
configuration or test an actual Claude Desktop/Code session.

## Tools

| Tool | Arguments | Result |
| --- | --- | --- |
| `staig_presets()` | None | Mapping of preset names to complete resolved configurations. `auto` is a runtime selector, not a stored preset. |
| `staig_run(...)` | `input_path`, `n_clusters=None`, `preset="auto"`, `batch_key=None`, `n_seeds=1`, `output_path=None`, `platform="auto"`, `bin_size=None`, `min_counts=None` | Metrics, output path, spot/domain counts, core keys written, elapsed seconds |
| `staig_extract_image_features(...)` | `input_path`, `image_path`, `output_npy` | `.npy` path, feature shape/dtype, row-order description and elapsed seconds |

Example `staig_run` arguments:

```json
{
  "input_path": "/absolute/path/to/section.h5ad",
  "n_clusters": 7,
  "preset": "auto",
  "n_seeds": 1,
  "output_path": "/absolute/path/to/section_domains.h5ad"
}
```

`input_path` can be any platform output recognised by `staig.read` (Space Ranger, Visium HD,
Xenium, MERSCOPE, CosMx, Stereo-seq, Slide-seq, seqFISH folders or files; see
[Input formats](input_formats.md)); `platform`, `bin_size` and `min_counts` are passed to the
reader. When `output_path` is omitted, the
output is a sibling `<input-stem>_staig.h5ad`. Existing paths, including the
input, are refused. `staig_run` uses the preset's full epoch count, automatic
image handling and standard refinement. It does not expose an epoch override;
use the [CLI examples](agent_guide.md#four-runnable-recipes) for a short pilot.

Input requirements and caveats:

- Expression defaults to raw counts. For already log-normalised `.h5ad`, first
  use `staig.pp.preprocess(adata, flavor="seurat", normalize=False, log1p=False)`
  and save the prepared `X_staig_input`. Do not rerun normalisation on that input.
- Several slices must already be concatenated (e.g. with `pp.concat_slices`) and
  saved, with `batch_key` identifying their non-missing slice labels. The tool
  accepts one path, not a list. Graphs must reflect that batch definition.
- Image extraction requires full-resolution coordinates and Visium
  `fiducial_diameter_fullres` metadata. Its output order matches the loaded
  `obs_names`; if `truth.txt` exists the Visium loader drops unlabelled spots.
  Load the same input with the same filtering before attaching the `.npy` via
  `staig.img.extract_image_features(adata, method="features.npy")`.
- The server performs real training, image extraction and evaluation. Large
  inputs, multi-seed consensus and full-data silhouettes can be expensive; the
  minimal server is intended for local bounded runs, not background scheduling.
  Choose client timeouts for the actual workload, or use the Python/CLI path.

## Tests

`tests/test_mcp_server.py` exercises the server with the SDK's in-memory client: tool discovery, a training
run through `staig_run` with its saved AnnData output, image extraction and the error responses.
