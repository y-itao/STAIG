"""Readers for platform-native outputs, checked on synthetic layouts that follow the vendor formats
(the real vendor sample files are exercised by tests/test_io_samples.py when STAIG_SAMPLES is set)."""

import gzip
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import staig
from staig import io, pp

rng = np.random.default_rng(0)
HAS_PYARROW = True
try:
    import pyarrow  # noqa: F401
except ImportError:                                 # the ``io`` extra is optional
    HAS_PYARROW = False


# ------------------------------------------------------------------------------ fixture writers
def _counts(n_cells, genes):
    return rng.poisson(2.0, size=(n_cells, len(genes))).astype(np.int32)


def write_10x_h5(path: Path, counts: np.ndarray, barcodes, genes, feature_types=None, genome="GRCh38"):
    """Cell Ranger / Space Ranger v3 HDF5 layout (matrix/{data,indices,indptr,shape,barcodes,features/*})."""
    x = sp.csc_matrix(counts.T)                        # stored genes x cells, column = cell
    feature_types = feature_types or ["Gene Expression"] * len(genes)
    with h5py.File(path, "w") as f:
        f.attrs["library_ids"] = np.array([b"library"])
        g = f.create_group("matrix")
        g.create_dataset("data", data=x.data.astype(np.int32))
        g.create_dataset("indices", data=x.indices.astype(np.int64))
        g.create_dataset("indptr", data=x.indptr.astype(np.int64))
        g.create_dataset("shape", data=np.array([len(genes), len(barcodes)], dtype=np.int32))
        g.create_dataset("barcodes", data=np.array(barcodes, dtype="S"))
        ft = g.create_group("features")
        ft.create_dataset("id", data=np.array([f"ENSG{i:08d}" for i in range(len(genes))], dtype="S"))
        ft.create_dataset("name", data=np.array(genes, dtype="S"))
        ft.create_dataset("feature_type", data=np.array(feature_types, dtype="S"))
        ft.create_dataset("genome", data=np.array([genome] * len(genes), dtype="S"))
        ft.create_dataset("_all_tag_keys", data=np.array(["genome"], dtype="S"))


def write_png(path: Path, shape=(20, 30)):
    from matplotlib.image import imsave

    imsave(path, rng.random((*shape, 3)))


def make_visium(root: Path, n=40, list_style=False, extra_feature=False):
    root.mkdir(parents=True)
    genes = [f"G{i}" for i in range(30)] + (["CD3_protein"] if extra_feature else [])
    barcodes = [f"BC{i:04d}-1" for i in range(n)]
    ft = ["Gene Expression"] * 30 + (["Antibody Capture"] if extra_feature else [])
    write_10x_h5(root / "filtered_feature_bc_matrix.h5", _counts(n, genes), barcodes, genes, ft)
    sdir = root / "spatial"
    sdir.mkdir()
    idx = np.arange(n)
    pos = pd.DataFrame({"barcode": barcodes, "in_tissue": 1, "array_row": idx // 8, "array_col": idx % 8,
                        "pxl_row_in_fullres": (idx // 8) * 100 + 50, "pxl_col_in_fullres": (idx % 8) * 100 + 50})
    if list_style:
        pos.to_csv(sdir / "tissue_positions_list.csv", header=False, index=False)
    else:
        pos.to_csv(sdir / "tissue_positions.csv", index=False)
    json.dump({"spot_diameter_fullres": 60.0, "fiducial_diameter_fullres": 97.0, "tissue_hires_scalef": 0.1,
               "tissue_lowres_scalef": 0.03}, open(sdir / "scalefactors_json.json", "w"))
    write_png(sdir / "tissue_hires_image.png")
    write_png(sdir / "tissue_lowres_image.png")
    return pos


def make_visium_hd(root: Path, n_side=6):
    pytest.importorskip("pyarrow")                  # Visium HD positions are parquet tables
    root.mkdir(parents=True)
    genes = [f"G{i}" for i in range(25)]
    out = {}
    for bin_um, prefix in ((2, "s_002um"), (8, "s_008um")):
        bdir = root / "binned_outputs" / f"square_{bin_um:03d}um"
        (bdir / "spatial").mkdir(parents=True)
        side = n_side * (8 // bin_um)
        rows, cols = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
        barcodes = [f"{prefix}_{r:05d}_{c:05d}-1" for r, c in zip(rows.ravel(), cols.ravel())]
        pos = pd.DataFrame({"barcode": barcodes, "in_tissue": 1, "array_row": rows.ravel(), "array_col": cols.ravel(),
                            "pxl_row_in_fullres": rows.ravel() * bin_um + bin_um / 2,
                            "pxl_col_in_fullres": cols.ravel() * bin_um + bin_um / 2})
        pos.loc[pos.index[:3], "in_tissue"] = 0
        pos.to_parquet(bdir / "spatial" / "tissue_positions.parquet")
        write_10x_h5(bdir / "filtered_feature_bc_matrix.h5", _counts(len(barcodes), genes), barcodes, genes)
        json.dump({"spot_diameter_fullres": float(bin_um), "bin_size_um": float(bin_um), "microns_per_pixel": 1.0,
                   "tissue_hires_scalef": 1.0, "tissue_lowres_scalef": 0.1, "fiducial_diameter_fullres": 300.0},
                  open(bdir / "spatial" / "scalefactors_json.json", "w"))
        write_png(bdir / "spatial" / "tissue_hires_image.png")
        out[bin_um] = pos
    # segmented cells: each cell = a 2x2 block of 2 um bins
    seg = root / "segmented_outputs"
    (seg / "spatial").mkdir(parents=True)
    pos2 = out[2]
    cell_of_bin = (pos2["array_row"] // 2) * 100 + pos2["array_col"] // 2
    cell_ids = {c: f"cellid_{i + 1:09d}-1" for i, c in enumerate(sorted(cell_of_bin.unique()))}
    mapping = pd.DataFrame({"square_002um": pos2["barcode"], "square_008um": "", "square_016um": "",
                            "cell_id": cell_of_bin.map(cell_ids), "in_nucleus": True, "in_cell": True})
    mapping.to_parquet(root / "barcode_mappings.parquet")
    cells = list(cell_ids.values())
    write_10x_h5(seg / "filtered_feature_cell_matrix.h5", _counts(len(cells), genes), cells, genes)
    json.dump({"microns_per_pixel": 1.0, "tissue_hires_scalef": 1.0},
              open(seg / "spatial" / "scalefactors_json.json", "w"))
    return out, cell_ids


def make_xenium(root: Path, n=30):
    root.mkdir(parents=True)
    genes = [f"G{i}" for i in range(20)] + ["NegControlProbe_1", "UnassignedCodeword_5", "BLANK_0001"]
    ft = ["Gene Expression"] * 20 + ["Negative Control Probe", "Unassigned Codeword", "Negative Control Codeword"]
    ids = [f"{''.join(rng.choice(list('abcdefghijklmnop'), 8))}-1" for _ in range(n)]
    counts = _counts(n, genes)
    counts[0] = 0                                       # an empty cell -> removed by min_counts
    write_10x_h5(root / "cell_feature_matrix.h5", counts, ids, genes, ft)
    cells = pd.DataFrame({"cell_id": ids, "x_centroid": rng.random(n) * 500, "y_centroid": rng.random(n) * 500,
                          "transcript_counts": counts[:, :20].sum(1), "cell_area": rng.random(n) * 50 + 20,
                          "nucleus_area": 10.0, "nucleus_count": 1, "segmentation_method": "Segmentation"})
    with gzip.open(root / "cells.csv.gz", "wt") as fh:
        cells.to_csv(fh, index=False)
    if HAS_PYARROW:
        cells.to_parquet(root / "cells.parquet")
    json.dump({"analysis_sw_version": "xenium-3.0.0.15", "pixel_size": 0.2125, "panel_name": "test", "num_cells": n,
               "region_name": "region"}, open(root / "experiment.xenium", "w"))
    tx = pd.DataFrame({"transcript_id": np.arange(200), "cell_id": rng.choice(ids[:10] + ["UNASSIGNED"], 200),
                       "feature_name": rng.choice(genes[:20] + ["NegControlProbe_1"], 200),
                       "x_location": rng.random(200) * 500, "y_location": rng.random(200) * 500, "qv": 40.0,
                       "is_gene": True})
    tx.loc[tx.feature_name == "NegControlProbe_1", "is_gene"] = False
    if HAS_PYARROW:
        tx.to_parquet(root / "transcripts.parquet")
    else:
        tx.to_csv(root / "transcripts.csv.gz", index=False)
    return cells


def make_merscope(root: Path, n=25, prefix="s1_"):
    root.mkdir(parents=True)
    genes = [f"G{i}" for i in range(15)] + ["Blank-1", "Blank-2"]
    ids = (np.arange(n) + 100000).astype(np.int64)
    df = pd.DataFrame(_counts(n, genes), columns=genes)
    df.insert(0, "cell", ids)
    df.to_csv(root / f"{prefix}cell_by_gene.csv", index=False)
    meta = pd.DataFrame({"EntityID": ids, "fov": np.arange(n) % 3, "volume": rng.random(n) * 1000,
                         "center_x": rng.random(n) * 1000, "center_y": rng.random(n) * 1000, "min_x": 0.0})
    meta.to_csv(root / f"{prefix}cell_metadata.csv", index=True)
    return df, meta


def make_cosmx(root: Path, n=24, prefix="Run1_"):
    root.mkdir(parents=True)
    genes = [f"G{i}" for i in range(12)] + ["NegPrb1", "NegPrb2", "SystemControl1"]
    fov = np.repeat([1, 2], n // 2)
    cell_id = np.tile(np.arange(1, n // 2 + 1), 2)
    df = pd.DataFrame(_counts(n, genes), columns=genes)
    df.insert(0, "cell_ID", cell_id)
    df.insert(0, "fov", fov)
    bg = pd.DataFrame([[1, 0] + [5] * len(genes)], columns=df.columns)   # background row cell_ID 0
    pd.concat([bg, df]).to_csv(root / f"{prefix}exprMat_file.csv", index=False)
    meta = pd.DataFrame({"fov": fov, "cell_ID": cell_id, "Area": 100, "AspectRatio": 1.0,
                         "CenterX_local_px": rng.random(n) * 4000, "CenterY_local_px": rng.random(n) * 4000,
                         "CenterX_global_px": rng.random(n) * 20000, "CenterY_global_px": rng.random(n) * 20000,
                         "Width": 10, "Height": 10, "Mean.PanCK": 1.0})
    meta.to_csv(root / f"{prefix}metadata_file.csv", index=False)
    pd.DataFrame({"fov": [1, 2], "x_global_px": [0, 4000], "y_global_px": [0, 0]}).to_csv(
        root / f"{prefix}fov_positions_file.csv", index=False)
    return df, meta


def make_gem(path: Path, cell_column=False, gz=False):
    genes = ["A", "B", "C"]
    rows = []
    for x in range(0, 8):
        for y in range(0, 8):
            for g in genes:
                rows.append((g, 100 + x, 200 + y, 1 + (x + y) % 3, (x // 4) * 2 + y // 4 + 1))
    df = pd.DataFrame(rows, columns=["geneID", "x", "y", "MIDCount", "CellID"])
    if not cell_column:
        df = df.drop(columns=["CellID"])
    header = ("#FileFormat=GEMv0.1\n#SortedBy=None\n#BinSize=1\n#StereoChip=SS200000135TL_D1\n"
              "#OffsetX=100\n#OffsetY=200\n")
    opener = gzip.open if gz else open
    with opener(path, "wt") as fh:
        fh.write(header)
        df.to_csv(fh, sep="\t", index=False)
    return df


def make_gef(path: Path):
    """Square-bin GEF after the SAW file-format manual (geneExp/bin1/{expression,gene})."""
    genes = ["A", "B", "C"]
    xs, ys, cnt, gidx = [], [], [], []
    for gi in range(len(genes)):
        for x in range(0, 6):
            for y in range(0, 6):
                xs.append(1000 + x)
                ys.append(2000 + y)
                cnt.append(1 + (x * y) % 4)
                gidx.append(gi)
    expr = np.zeros(len(xs), dtype=[("x", "<i4"), ("y", "<i4"), ("count", "<u2")])
    expr["x"], expr["y"], expr["count"] = xs, ys, cnt
    gene = np.zeros(len(genes), dtype=[("gene", "S32"), ("offset", "<u4"), ("count", "<u4")])
    per = len(xs) // len(genes)
    gene["gene"] = [g.encode() for g in genes]
    gene["offset"] = np.arange(len(genes)) * per
    gene["count"] = per
    with h5py.File(path, "w") as f:
        f.attrs["version"] = 2
        g = f.create_group("geneExp/bin1")
        d = g.create_dataset("expression", data=expr)
        for key, val in (("minX", 1000), ("minY", 2000), ("maxX", 1005), ("maxY", 2005), ("resolution", 500)):
            d.attrs[key] = val
        g.create_dataset("gene", data=gene)
    return pd.DataFrame({"gene": np.array(genes)[gidx], "x": xs, "y": ys, "count": cnt})


def make_cellbin_gef(path: Path):
    genes = ["A", "B", "C", "D"]
    n_cells = 5
    cell = np.zeros(n_cells, dtype=[("id", "<u4"), ("x", "<i4"), ("y", "<i4"), ("offset", "<u4"), ("geneCount", "<u2"),
                                    ("expCount", "<u2"), ("dnbCount", "<u2"), ("area", "<u2"), ("cellTypeID", "<u4"),
                                    ("clusterID", "<u4")])
    exp_rows = []
    offset = 0
    expected = np.zeros((n_cells, len(genes)))
    for i in range(n_cells):
        picks = [(i % 4), ((i + 1) % 4)]
        cell[i]["id"], cell[i]["x"], cell[i]["y"] = i, 10 * i, 20 * i
        cell[i]["offset"], cell[i]["geneCount"] = offset, len(picks)
        for g in picks:
            exp_rows.append((g, 3 + i))
            expected[i, g] = 3 + i
        offset += len(picks)
    exp = np.array(exp_rows, dtype=[("geneID", "<u4"), ("count", "<u2")])
    gene = np.zeros(len(genes), dtype=[("gene", "S32"), ("offset", "<u4"), ("cellCount", "<u4"), ("expCount", "<u4")])
    gene["gene"] = [g.encode() for g in genes]
    with h5py.File(path, "w") as f:
        f.attrs["resolution"] = 500
        g = f.create_group("cellBin")
        g.create_dataset("cell", data=cell)
        g.create_dataset("cellExp", data=exp)
        g.create_dataset("gene", data=gene)
    return expected


def make_slide_seq(root: Path, n=30, curio=False):
    root.mkdir(parents=True)
    genes = [f"G{i}" for i in range(18)]
    beads = ["".join(rng.choice(list("ACGT"), 14)) for _ in range(n)]
    counts = _counts(n, genes)
    loc = pd.DataFrame({"barcodes": beads, "xcoord": rng.random(n) * 5000, "ycoord": rng.random(n) * 5000})
    if curio:
        from scipy.io import mmwrite

        mmwrite(root / "sample_MoleculesPerMatchedBead.mtx", sp.csr_matrix(counts.T))
        pd.Series(beads).to_csv(root / "sample_barcodes.tsv", sep="\t", index=False, header=False)
        pd.Series(genes).to_csv(root / "sample_genes.tsv", sep="\t", index=False, header=False)
        loc.rename(columns={"barcodes": "barcode", "xcoord": "x", "ycoord": "y"}).to_csv(
            root / "sample_MatchedBeadLocation.csv", index=False)
    else:
        dge = pd.DataFrame(counts.T, index=pd.Index(genes, name="GENE"), columns=beads)
        with gzip.open(root / "Puck_200127_15.digital_expression.txt.gz", "wt") as fh:
            dge.to_csv(fh, sep="\t")
        loc.to_csv(root / "Puck_200127_15_bead_locations.csv")
    return counts, loc


def make_seqfish(root: Path, n=20):
    root.mkdir(parents=True)
    genes = [f"G{i}" for i in range(10)]
    df = pd.DataFrame(_counts(n, genes), columns=genes, index=pd.Index(np.arange(1, n + 1)))
    df.to_csv(root / "Roi1_CellxGene.csv")
    coords = pd.DataFrame({"label": np.arange(1, n + 1), "area": 50.0, "center_x": rng.random(n) * 100,
                           "center_y": rng.random(n) * 100})
    coords.to_csv(root / "Roi1_CellCoordinates.csv", index=False)
    return df, coords


# ------------------------------------------------------------------------------------------ tests
def test_visium_layouts_and_antibody_features(tmp_path):
    pos = make_visium(tmp_path / "v1", extra_feature=True)
    a = io.read(tmp_path / "v1")
    assert a.uns["staig"]["platform"] == "visium" and a.n_vars == 30 and "spatial" in a.uns
    assert np.allclose(a.obsm["spatial"][0], [pos.pxl_col_in_fullres[0], pos.pxl_row_in_fullres[0]])
    make_visium(tmp_path / "v2", list_style=True)
    b = io.read(tmp_path / "v2" / "filtered_feature_bc_matrix.h5")                 # a file path also works
    assert b.n_obs == 40 and io.detect_platform(tmp_path / "v2") == "visium"
    assert staig.auto_preset(a) == "visium_no_image"


def test_visium_hd_bins_and_cells(tmp_path):
    pos, cell_ids = make_visium_hd(tmp_path / "hd")
    a = io.read(tmp_path / "hd")                                                    # default 8 um bins
    assert a.uns["staig"]["platform"] == "visium_hd" and a.uns["staig"]["bin_size_um"] == 8.0
    assert a.n_obs == 36 - 3 and (a.obs["in_tissue"] == 1).all()                    # in_tissue filter
    assert a.obsm["spatial"][0].tolist() == [pos[8].pxl_col_in_fullres[3], pos[8].pxl_row_in_fullres[3]]
    b = io.read(tmp_path / "hd" / "binned_outputs" / "square_002um")
    assert b.n_obs == 24 * 24 - 3 and b.uns["staig"]["bin_size_um"] == 2.0
    with pytest.raises(FileNotFoundError, match="available bin folders"):
        io.read(tmp_path / "hd", bin_size=16)
    c = io.read(tmp_path / "hd", level="cells")
    assert c.n_obs == len(cell_ids) and c.uns["staig"]["level"] == "cells"
    first = c.obsm["spatial"][list(c.obs_names).index("cellid_000000001-1")]
    assert np.allclose(first, [2.0, 2.0])                                           # mean of the four 2 um bins
    assert staig.auto_preset(a) == "stereo_seq"


def test_xenium_controls_and_qc(tmp_path):
    cells = make_xenium(tmp_path / "xe")
    a = io.read(tmp_path / "xe")
    assert a.uns["staig"]["platform"] == "xenium" and a.n_vars == 20
    assert a.uns["staig"]["dropped_features"] == {"Negative Control Probe": 1, "Unassigned Codeword": 1,
                                                  "Negative Control Codeword": 1}
    assert a.n_obs == 29 and a.uns["staig"]["qc"]["n_removed"] == 1                 # the empty cell
    assert a.uns["staig"]["xenium"]["pixel_size"] == 0.2125 and "cell_area" in a.obs
    row = cells.set_index("cell_id").loc[a.obs_names[0]]
    assert np.allclose(a.obsm["spatial"][0], [row.x_centroid, row.y_centroid])
    tx_file = "transcripts.parquet" if HAS_PYARROW else "transcripts.csv.gz"
    t = io.read_transcripts(tmp_path / "xe" / tx_file, min_counts=1)
    assert "NegControlProbe_1" not in t.var_names and t.n_obs <= 10 and "UNASSIGNED" not in t.obs_names
    assert staig.auto_preset(a) == "merfish"
    assert io.preprocess_defaults(a) == {"n_top_genes": None}


def test_merscope(tmp_path):
    df, meta = make_merscope(tmp_path / "mer")
    a = io.read(tmp_path / "mer", min_counts=0)
    assert a.uns["staig"]["platform"] == "merscope" and a.n_vars == 15 and "Blank-1" not in a.var_names
    assert a.obs["control_counts"].iloc[0] == df.iloc[0][["Blank-1", "Blank-2"]].sum()
    by_id = meta.set_index(meta.EntityID.astype(str)).loc[a.obs_names]
    assert np.allclose(a.obsm["spatial"][:, 0], by_id.center_x) and "volume" in a.obs
    b = io.read(tmp_path / "mer")
    assert b.n_obs <= a.n_obs and b.uns["staig"]["qc"]["min_counts"] == 20
    assert io.detect_platform(tmp_path / "mer") == "merscope"


def test_cosmx(tmp_path):
    df, meta = make_cosmx(tmp_path / "cos")
    a = io.read(tmp_path / "cos", min_counts=0)
    assert a.uns["staig"]["platform"] == "cosmx" and a.n_vars == 12 and a.n_obs == 24     # background row dropped
    assert list(a.obs_names[:2]) == ["1_1", "1_2"]
    assert a.obs["control_counts"].iloc[0] == df.iloc[0][["NegPrb1", "NegPrb2", "SystemControl1"]].sum()
    assert np.allclose(a.obsm["spatial"][:, 0], meta.CenterX_global_px)
    b = io.read_cosmx(tmp_path / "cos", coordinates="local", min_counts=0)
    assert np.allclose(b.obsm["spatial"][:, 1], meta.CenterY_local_px)
    (tmp_path / "cos" / "sub").mkdir()
    make_cosmx(tmp_path / "cos" / "sub" / "x", prefix="Run2_")
    with pytest.raises(FileNotFoundError, match="exactly one"):
        io.read(tmp_path / "cos")


def test_stereo_seq_gem_bins_and_cells(tmp_path):
    df = make_gem(tmp_path / "a.gem.gz", gz=True)
    a = io.read(tmp_path / "a.gem.gz", bin_size=4, min_counts=0)
    assert a.uns["staig"]["platform"] == "stereo_seq" and a.uns["staig"]["level"] == "bins"
    assert a.n_obs == 4 and a.n_vars == 3
    expected = df.groupby([(df.x - 100) // 4, (df.y - 200) // 4]).MIDCount.sum().to_numpy()
    assert np.allclose(np.sort(np.asarray(a.X.sum(1)).ravel()), np.sort(expected))
    centres = [(102.0, 202.0), (102.0, 206.0), (106.0, 202.0), (106.0, 206.0)]
    assert sorted(map(tuple, a.obsm["spatial"].tolist())) == centres
    make_gem(tmp_path / "c.gem", cell_column=True)
    c = io.read(tmp_path / "c.gem", min_counts=0)
    assert c.uns["staig"]["level"] == "cells" and c.n_obs == 4
    b = io.read(tmp_path / "c.gem", cellbin=False, bin_size=8, min_counts=0)
    assert b.n_obs == 2 and b.uns["staig"]["level"] == "bins"           # x 100-103 | 104-107 on the absolute grid
    # a plain .tsv with a GEM header is recognised as Stereo-seq
    make_gem(tmp_path / "plain.tsv")
    assert io.detect_platform(tmp_path / "plain.tsv") == "stereo_seq"


def test_stereo_seq_gef(tmp_path):
    truth = make_gef(tmp_path / "x.gef")
    a = io.read(tmp_path / "x.gef", bin_size=3, min_counts=0)
    assert a.n_obs == 9 and a.n_vars == 3                                # absolute grid: 1000..1005 -> 3 bins per axis
    exp = truth.groupby([truth.x // 3, truth.y // 3])["count"].sum().to_numpy()
    assert np.allclose(np.sort(np.asarray(a.X.sum(1)).ravel()), np.sort(exp))
    b = io.read(tmp_path / "x.gef", bin_size=1, min_counts=0)
    assert b.n_obs == 36 and np.allclose(b.obsm["spatial"].min(0), [1000.5, 2000.5])
    expected = make_cellbin_gef(tmp_path / "cells.cellbin.gef")
    c = io.read(tmp_path / "cells.cellbin.gef", min_counts=0)
    assert c.uns["staig"]["level"] == "cells" and np.allclose(c.X.toarray(), expected)
    assert c.obsm["spatial"][2].tolist() == [20.0, 40.0]


def test_slide_seq_dge_and_curio(tmp_path):
    counts, loc = make_slide_seq(tmp_path / "puck")
    a = io.read(tmp_path / "puck", min_counts=0)
    assert a.uns["staig"]["platform"] == "slide_seq" and a.shape == counts.shape
    i = list(loc.barcodes).index(a.obs_names[0])
    assert np.allclose(a.obsm["spatial"][0], [loc.xcoord[i], loc.ycoord[i]])
    assert np.allclose(a.X[0].toarray().ravel(), counts[i])
    counts2, loc2 = make_slide_seq(tmp_path / "curio", curio=True)
    b = io.read(tmp_path / "curio", min_counts=0)
    assert b.shape == counts2.shape and b.uns["staig"]["platform"] == "slide_seq"
    assert staig.auto_preset(a) == "slide_seq"


def test_seqfish_and_generic_tables(tmp_path):
    df, coords = make_seqfish(tmp_path / "sf")
    a = io.read(tmp_path / "sf", min_counts=0)
    assert a.uns["staig"]["platform"] == "seqfish" and a.shape == df.shape
    assert np.allclose(a.obsm["spatial"][:, 0], coords.center_x) and a.uns["staig"]["coordinate_unit"] == "micron"
    # generic: genes x cells table + coordinates with other column names
    (tmp_path / "gen").mkdir()
    df.T.to_csv(tmp_path / "gen" / "expr.tsv", sep="\t")
    renamed = coords.rename(columns={"label": "cell", "center_x": "X", "center_y": "Y"})
    renamed.to_csv(tmp_path / "gen" / "pos.csv", index=False)
    b = io.read_table(tmp_path / "gen" / "expr.tsv", tmp_path / "gen" / "pos.csv")
    assert b.shape == df.shape and np.allclose(b.X.toarray(), df.to_numpy())
    # coordinates inside the counts table
    full = df.copy()
    full["x"], full["y"] = coords.center_x.to_numpy(), coords.center_y.to_numpy()
    full.to_csv(tmp_path / "gen" / "with_xy.csv")
    c = io.read_table(tmp_path / "gen" / "with_xy.csv")
    assert c.n_vars == 10 and np.allclose(c.obsm["spatial"][:, 1], coords.center_y)


def test_transcript_table_with_counts_and_fov(tmp_path):
    tx = pd.DataFrame({"fov": [1, 1, 2, 2, 2, 2], "cell_ID": [1, 1, 1, 1, 0, 1],
                       "target": ["A", "B", "A", "A", "A", "NegPrb1"],
                       "x_global_px": [0, 2, 10, 12, 50, 11], "y_global_px": [0, 2, 10, 12, 50, 11]})
    tx.to_csv(tmp_path / "S_tx_file.csv", index=False)
    a = io.read_transcripts(tmp_path / "S_tx_file.csv", min_counts=0)
    assert sorted(a.obs_names) == ["1_1", "2_1"] and list(a.var_names) == ["A", "B"]
    assert a[a.obs_names == "2_1"].X.toarray().tolist() == [[2.0, 0.0]]
    assert np.allclose(a.obsm["spatial"][list(a.obs_names).index("2_1")], [11, 11])


def test_h5ad_with_obs_coordinates_and_read_lists(tmp_path):
    counts = _counts(12, list("ABCD"))
    a = ad.AnnData(X=counts.astype(np.float32))
    a.obs["x"], a.obs["y"] = rng.random(12), rng.random(12)
    a.write_h5ad(tmp_path / "a.h5ad")
    b = io.read(tmp_path / "a.h5ad")
    assert "spatial" in b.obsm and b.uns["staig"]["platform"] == "h5ad"
    make_visium(tmp_path / "s1")
    make_visium(tmp_path / "s2")
    parts = io.read([tmp_path / "s1", tmp_path / "s2"])
    assert [p.obs["slice"].iloc[0] for p in parts] == ["s1", "s2"]
    parts = io.read({"first": tmp_path / "s1", "second": tmp_path / "s2"})
    assert parts[1].obs["slice"].iloc[0] == "second"


def test_detect_platform_errors(tmp_path):
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "notes.txt").write_text("hello\n")
    with pytest.raises(ValueError, match="Supported layouts"):
        io.detect_platform(tmp_path / "empty")
    with pytest.raises(ValueError, match="platform must be one of"):
        io.read(tmp_path / "empty", platform="nope")


def test_run_from_paths_and_integration(tmp_path):
    make_visium(tmp_path / "s1", n=60)
    make_visium(tmp_path / "s2", n=60)
    m = staig.run(tmp_path / "s1", n_clusters=2, epochs=2, device="cpu", verbose=False, hidden_dim=8, proj_dim=8,
                  n_seeds=1, image=None, n_neighbors=3, cluster_method="kmeans")
    assert m.adata.uns["staig"]["platform"] == "visium" and m.adata.obs["staig_domain"].nunique() == 2
    assert m.adata.uns["staig"]["preprocess"]["n_top_genes"] == 30
    m2 = staig.run([tmp_path / "s1", tmp_path / "s2"], n_clusters=2, epochs=2, device="cpu", verbose=False,
                   hidden_dim=8, proj_dim=8, n_seeds=1, n_neighbors=3, read_kwargs={"min_counts": 0},
                   cluster_method="kmeans")
    assert m2.adata.n_obs == 120 and m2.adata.obs["slice"].nunique() == 2
    assert m2.adata.uns["staig"]["config"]["within_slice_negatives"]
    # targeted panel from a path: every gene is used
    make_xenium(tmp_path / "xe", n=60)
    m3 = staig.run(tmp_path / "xe", n_clusters=2, epochs=2, device="cpu", verbose=False, hidden_dim=8, proj_dim=8,
                   n_seeds=1, n_neighbors=3, cluster_method="kmeans")
    assert m3.adata.uns["staig"]["preprocess"]["n_top_genes"] == 20
    assert m3.config.edge_removal == "fixed"


def test_cli_inspect_and_run_on_platform_folder(tmp_path, capsys):
    from staig import cli

    make_visium(tmp_path / "s1", n=60)
    cli.main(["inspect", str(tmp_path / "s1")])
    out = capsys.readouterr().out
    assert "platform=visium" in out and "60 observations" in out
    cfg = staig.STAIGConfig(epochs=2, hidden_dim=8, proj_dim=8, device="cpu", verbose=False, n_neighbors=3)
    cfg.to_yaml(tmp_path / "c.yaml")
    cli.main(["run", str(tmp_path / "s1"), "--n-clusters", "2", "--n-seeds", "1", "--config",
              str(tmp_path / "c.yaml"), "--cluster", "kmeans", "--output", str(tmp_path / "out.h5ad")])
    b = ad.read_h5ad(tmp_path / "out.h5ad")
    assert b.obs["staig_domain"].nunique() == 2 and b.uns["staig"]["platform"] == "visium"


# ===================================================================== edge cases of the readers
def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_cosmx_transcript_cells_keyed_by_fov_even_with_text_ids(tmp_path):
    p = _write(tmp_path / "S_tx_file.csv",
               "fov,cell_ID,target,x_global_px,y_global_px\n1,1,A,0,0\n2,1,A,100,0\n2,UNASSIGNED,A,5,5\n")
    a = io.read_transcripts(p, min_counts=0)
    assert sorted(a.obs_names) == ["1_1", "2_1"] and a.X.toarray().tolist() == [[1.0], [1.0]]
    assert np.allclose(a.obsm["spatial"][list(a.obs_names).index("2_1")], [100, 0])
    assert a.uns["staig"]["qc"]["removed"]["unassigned"] == 1


def test_list_input_with_preprocess_false_keeps_prepared_input():
    raw = np.tile([1, 2, 3, 4], (8, 1)).astype(np.float32)
    parts = []
    for i in range(2):
        a = ad.AnnData(X=raw.copy())
        a.obsm["spatial"] = np.arange(16, dtype=float).reshape(8, 2) + i
        a.obsm["X_staig_input"] = np.ones((8, 4), np.float32) * (i + 1)
        parts.append(a)
    m = staig.run(parts, preprocess=False, image=None, cluster_method=None, device="cpu", n_seeds=1, epochs=1,
                  verbose=False, hidden_dim=4, proj_dim=4, n_neighbors=2)
    x = m.adata.obsm["X_staig_input"]
    assert x.shape == (16, 4) and x[0, 0] == 1 and x[8, 0] == 2 and m.adata.X[0, 0] == 1       # untouched
    assert list(m.adata.obs["slice"].cat.categories) == ["0", "1"]
    with pytest.raises(ValueError, match="X_staig_input"):
        staig.run([ad.AnnData(X=raw.copy(), obsm={"spatial": parts[0].obsm["spatial"]}), parts[1]], preprocess=False,
                  image=None, cluster_method=None, device="cpu")


def test_cli_detects_normalised_h5ad_in_lists_and_rejects_conflicting_flags(tmp_path, monkeypatch):
    from staig import cli

    a = ad.AnnData(X=np.array([[0.2, 0.4], [0.5, 0.8], [0.1, 0.9]], np.float32))
    a.obsm["spatial"] = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    a.write_h5ad(tmp_path / "a.h5ad")
    a.write_h5ad(tmp_path / "b.h5ad")
    seen = {}

    def fake_run(adata, **kw):
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr("staig.run", fake_run)
    with pytest.raises(RuntimeError):
        cli.main(["run", str(tmp_path / "a.h5ad"), str(tmp_path / "b.h5ad")])
    assert seen["preprocess"] == {"normalize": False, "log1p": False}
    with pytest.raises(SystemExit):
        cli.main(["run", str(tmp_path / "a.h5ad"), "--counts", "--log-normalized"])


def test_gef_offsets_are_honoured(tmp_path):
    expr = np.zeros(2, dtype=[("x", "<i4"), ("y", "<i4"), ("count", "<u2")])
    expr["x"], expr["y"], expr["count"] = [5, 5], [7, 7], [3, 7]
    gene = np.zeros(2, dtype=[("gene", "S32"), ("offset", "<u4"), ("count", "<u4")])
    gene["gene"], gene["offset"], gene["count"] = [b"A", b"B"], [1, 0], [1, 1]        # reversed offsets
    with h5py.File(tmp_path / "o.gef", "w") as f:
        g = f.create_group("geneExp/bin1")
        d = g.create_dataset("expression", data=expr)
        d.attrs["resolution"] = 500
        g.create_dataset("gene", data=gene)
    a = io.read(tmp_path / "o.gef", bin_size=1, min_counts=0)
    assert a.X.toarray().tolist() == [[7.0, 3.0]]                                       # A <- row 1, B <- row 0
    gene["offset"] = [0, 0]
    with h5py.File(tmp_path / "bad.gef", "w") as f:
        g = f.create_group("geneExp/bin1")
        g.create_dataset("expression", data=expr)
        g.create_dataset("gene", data=gene)
    with pytest.raises(ValueError, match="offset"):
        io.read(tmp_path / "bad.gef", bin_size=1, min_counts=0)


def test_constant_matrix_does_not_enter_the_loess_fit():
    a = ad.AnnData(np.c_[np.tile(np.arange(1, 5, dtype=np.float32), (12, 1)), np.zeros((12, 16), np.float32)])
    pp.preprocess(a, n_top_genes=2)
    info = a.uns["staig"]["preprocess"]
    assert info["flavor"] == "all" and "0 genes vary" in info["hvg_note"] and a.var["highly_variable"].sum() == 0


def test_visium_without_images_keeps_coordinates(tmp_path):
    make_visium(tmp_path / "v")
    a = io.read(tmp_path / "v", load_images=False)
    assert a.obsm["spatial"].shape == (40, 2) and "spatial" in a.uns
    lib = next(iter(a.uns["spatial"]))
    assert a.uns["spatial"][lib]["images"] == {} and "spot_diameter_fullres" in a.uns["spatial"][lib]["scalefactors"]


def test_detection_and_readers_agree_on_nested_and_file_paths(tmp_path):
    pytest.importorskip("pyarrow")
    make_visium_hd(tmp_path / "proj" / "outs")
    assert io.detect_platform(tmp_path / "proj") == "visium_hd"
    assert io.read(tmp_path / "proj").n_obs == 33
    h5 = tmp_path / "proj" / "outs" / "binned_outputs" / "square_008um" / "filtered_feature_bc_matrix.h5"
    assert io.detect_platform(h5) == "visium_hd" and io.read(h5).n_obs == 33
    make_visium(tmp_path / "vis" / "outs")
    assert io.read(tmp_path / "vis").n_obs == 40
    (tmp_path / "only").mkdir()
    a = ad.AnnData(X=np.ones((3, 2), np.float32))
    a.obsm["spatial"] = np.zeros((3, 2))
    a.write_h5ad(tmp_path / "only" / "a.h5ad")
    assert io.detect_platform(tmp_path / "only") == "h5ad" and io.read(tmp_path / "only").n_obs == 3
    make_xenium(tmp_path / "xe")
    (tmp_path / "xe" / "experiment.xenium").unlink()
    (tmp_path / "xe" / "cell_feature_matrix.h5").rename(tmp_path / "xe" / "cfm.h5")
    import scanpy as sc

    mtx = sc.read_10x_h5(tmp_path / "xe" / "cfm.h5", gex_only=False)
    from scipy.io import mmwrite

    d = tmp_path / "xe" / "cell_feature_matrix"
    d.mkdir()
    with gzip.open(d / "matrix.mtx.gz", "wb") as fh:
        mmwrite(fh, sp.csc_matrix(mtx.X.T))
    pd.Series(mtx.obs_names).to_csv(d / "barcodes.tsv.gz", sep="\t", index=False, header=False)
    pd.DataFrame({"id": mtx.var["gene_ids"], "name": mtx.var_names, "type": mtx.var["feature_types"]}).to_csv(
        d / "features.tsv.gz", sep="\t", index=False, header=False)
    assert io.detect_platform(tmp_path / "xe") == "xenium" and io.read(tmp_path / "xe").n_vars == 20


def test_geojson_area_centroids(tmp_path):
    square = [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]
    subdivided = [[[0, 0], [5, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]            # extra vertex on one edge
    with_hole = [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], [[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]]]
    g = {"type": "FeatureCollection", "features": [
        {"properties": {"cell_id": 1}, "geometry": {"type": "Polygon", "coordinates": square}},
        {"properties": {"cell_id": 2}, "geometry": {"type": "Polygon", "coordinates": subdivided}},
        {"properties": {"cell_id": 3}, "geometry": {"type": "Polygon", "coordinates": with_hole}},
        {"properties": {"cell_id": 4},
         "geometry": {"type": "MultiPolygon",
                      "coordinates": [square, [[[20, 0], [22, 0], [22, 2], [20, 2], [20, 0]]]]}},
    ]}
    json.dump(g, open(tmp_path / "cells.geojson", "w"))
    c = io._geojson_centroids(tmp_path / "cells.geojson")
    assert np.allclose(c.loc["cellid_000000001-1"], [5, 5]) and np.allclose(c.loc["cellid_000000002-1"], [5, 5])
    hole_centroid = (100 * np.array([5, 5]) - 4 * np.array([2, 2])) / 96
    assert np.allclose(c.loc["cellid_000000003-1"], hole_centroid)
    assert np.allclose(c.loc["cellid_000000004-1"], (100 * np.array([5, 5]) + 4 * np.array([21, 1])) / 104)


def test_identifiers_keep_leading_zeros_and_duplicates_are_handled(tmp_path):
    _write(tmp_path / "expr.csv", "gene,001,002\nA,4,5\nB,6,7\n")
    _write(tmp_path / "pos.csv", "cell,x,y\n001,10,20\n002,30,40\n002,99,88\n")
    with pytest.warns(UserWarning, match="duplicated ids"):
        a = io.read_table(tmp_path / "expr.csv", tmp_path / "pos.csv")
    assert list(a.obs_names) == ["001", "002"] and a.X.toarray().tolist() == [[4, 6], [5, 7]]
    assert np.allclose(a.obsm["spatial"], [[10, 20], [30, 40]])
    _write(tmp_path / "dup.csv", "cell,A,B\nc,1,2\nc,3,4\n")
    with pytest.raises(ValueError, match="duplicated observation ids"):
        io.read_table(tmp_path / "dup.csv", tmp_path / "pos.csv")
    make_merscope(tmp_path / "mer", n=3)
    m = io.read(tmp_path / "mer", min_counts=0)
    assert all(len(n) == 6 for n in m.obs_names)                                      # "100000" ... as written


def test_background_and_is_gene_are_explicit(tmp_path):
    _write(tmp_path / "t.csv", "cell_id,gene,x,y\n0,A,0,0\n1,A,1,1\n,A,5,5\n")
    a = io.read_transcripts(tmp_path / "t.csv", min_counts=0)
    assert list(a.obs_names) == ["1"]
    _write(tmp_path / "g.csv", "cell_id,gene,x,y,is_gene\n1,A,0,0,true\n1,B,0,0,false\n1,C,0,0,unknown\n")
    with pytest.raises(ValueError, match="boolean"):
        io.read_transcripts(tmp_path / "g.csv", min_counts=0)
    _write(tmp_path / "ok.csv", "cell_id,gene,x,y,is_gene\n1,A,0,0,true\n1,B,0,0,false\n")
    b = io.read_transcripts(tmp_path / "ok.csv", min_counts=0)
    assert list(b.var_names) == ["A"] and b.obs["control_counts"].iloc[0] == 1
    _write(tmp_path / "cells.gem", "geneID\tx\ty\tUMICount\tlabel\nA\t1\t1\t2\t0\nA\t2\t2\t3\t1\nA\t3\t3\t4\t\n")
    c = io.read(tmp_path / "cells.gem", min_counts=0)
    assert list(c.obs_names) == ["1"] and c.uns["staig"]["qc"]["removed"]["background"] == 2


def test_transcript_coordinates_prefer_global_and_reject_local_multi_fov(tmp_path):
    _write(tmp_path / "v.csv", "cell_id,gene,x,y,global_x,global_y\n1,A,1,2,101,102\n2,A,1,2,501,502\n")
    a = io.read_transcripts(tmp_path / "v.csv", min_counts=0)
    assert np.allclose(a.obsm["spatial"], [[101, 102], [501, 502]])
    _write(tmp_path / "c.csv", "fov,cell_ID,target,x_local_px,y_local_px\n1,1,A,0,0\n2,1,A,0,0\n")
    with pytest.raises(ValueError, match="per-FOV"):
        io.read_transcripts(tmp_path / "c.csv", min_counts=0)
    _write(tmp_path / "one.csv", "fov,cell_ID,target,x_local_px,y_local_px\n1,1,A,3,4\n1,2,A,5,6\n")
    b = io.read_transcripts(tmp_path / "one.csv", min_counts=0)
    assert b.uns["staig"]["coordinate_unit"] == "local"


def test_integration_defaults_are_order_independent_and_names_unique(tmp_path):
    make_visium(tmp_path / "a" / "outs", n=50)
    make_visium(tmp_path / "b" / "outs", n=50)
    make_xenium(tmp_path / "xe", n=50)
    parts = io.read([tmp_path / "a" / "outs", tmp_path / "b" / "outs"])
    assert [p.obs["slice"].iloc[0] for p in parts] == ["a", "b"]
    kw = dict(n_clusters=2, epochs=1, device="cpu", verbose=False, hidden_dim=4, proj_dim=4, n_seeds=1,
              n_neighbors=3, cluster_method="kmeans", read_kwargs={"min_counts": 0})
    m1 = staig.run([tmp_path / "xe", tmp_path / "a" / "outs"], **kw)
    m2 = staig.run([tmp_path / "a" / "outs", tmp_path / "xe"], **kw)
    assert m1.adata.uns["staig"]["platform"] == "mixed" and m1.adata.n_vars == m2.adata.n_vars
    assert m1.adata.uns["staig"]["concat"]["n_common_hvg"] == m2.adata.uns["staig"]["concat"]["n_common_hvg"]
    m3 = staig.run({"first": tmp_path / "a" / "outs", "second": tmp_path / "b" / "outs"}, batch_key="sample", **kw)
    assert list(m3.adata.obs["sample"].cat.categories) == ["first", "second"]
    with pytest.warns(UserWarning, match="does not take"):
        io.read(tmp_path / "xe", bin_size=8)                                            # ignored, not an error


def test_concat_uses_the_hvg_recovery(monkeypatch):
    import scanpy as sc
    from test_pipeline import make_synthetic

    a = make_synthetic(seed=5, n_genes=120)
    original = sc.pp.highly_variable_genes
    calls = {"n": 0}

    def flaky(adata, **kw):
        calls["n"] += 1
        if kw.get("flavor") == "seurat_v3" and calls["n"] == 1:
            raise ValueError(b"There are other near singularities as well. 0.03")
        return original(adata, **kw)

    monkeypatch.setattr(sc.pp, "highly_variable_genes", flaky)
    merged = pp.concat_slices([a, a.copy()], batch_categories=["x", "y"], n_top_genes=30)
    assert merged.obsm["X_staig_input"].shape[1] == 30


def test_image_auto_only_for_visium_and_explicit_image_rejected_for_lists(tmp_path, monkeypatch):
    make_xenium(tmp_path / "xe", n=40)
    calls = []
    monkeypatch.setattr("staig.img.patches.resolve_image_path", lambda adata, *a, **k: "pretend.tif")

    def spy(adata, image=None, **kw):
        calls.append(image)
        raise RuntimeError("should not be called")

    monkeypatch.setattr("staig.img.extract_image_features", spy)
    staig.run(tmp_path / "xe", n_clusters=2, epochs=1, device="cpu", verbose=False, hidden_dim=4, proj_dim=4, n_seeds=1,
              n_neighbors=3, cluster_method="kmeans")
    assert calls == []
    with pytest.raises(ValueError, match="single input"):
        staig.run([tmp_path / "xe", tmp_path / "xe"], image="explicit.tif", device="cpu")


def test_control_provenance_for_visium_mtx_and_transcripts(tmp_path):
    make_visium(tmp_path / "v", extra_feature=True)
    a = io.read(tmp_path / "v")
    assert a.uns["staig"]["dropped_features"] == {"Antibody Capture": 1} and "control_counts" in a.obs
    b = io.read(tmp_path / "v", drop_controls=False)
    assert b.n_vars == 31
    from scipy.io import mmwrite

    (tmp_path / "mtx").mkdir()
    mmwrite(tmp_path / "mtx" / "sample_counts.mtx", sp.csr_matrix(np.array([[4, 9], [2, 1]])).T)
    pd.Series(["c1", "c2"]).to_csv(tmp_path / "mtx" / "sample_barcodes.tsv", sep="\t", index=False, header=False)
    pd.Series(["A", "NegPrb1"]).to_csv(tmp_path / "mtx" / "sample_genes.tsv", sep="\t", index=False, header=False)
    _write(tmp_path / "mtx" / "pos.csv", "cell,x,y\nc1,0,0\nc2,1,1\n")
    c = io.read_table(tmp_path / "mtx" / "sample_counts.mtx", tmp_path / "mtx" / "pos.csv")
    assert list(c.var_names) == ["A"] and c.obs["control_counts"].tolist() == [9, 1]
    _write(tmp_path / "t.csv", "cell_id,gene,x,y\nc,A,0,0\nc,NegPrb1,0,0\n")
    d = io.read_transcripts(tmp_path / "t.csv", min_counts=0)
    assert list(d.var_names) == ["A"] and d.obs["control_counts"].iloc[0] == 1


def test_empty_results_and_invalid_counts_fail_clearly(tmp_path):
    _write(tmp_path / "e.csv", "cell,A,B\nc1,1,0\nc2,0,1\n")
    _write(tmp_path / "p.csv", "cell,x,y\nc1,,0\nc2,1,1\n")
    with pytest.raises(ValueError, match="no observations left"):
        with pytest.warns(UserWarning):
            io.read_table(tmp_path / "e.csv", tmp_path / "p.csv", min_counts=100)
    _write(tmp_path / "neg.csv", "cell,A,B\nc1,-2,5\nc2,0,1\n")
    _write(tmp_path / "pos.csv", "cell,x,y\nc1,0,0\nc2,1,1\n")
    with pytest.raises(ValueError, match="negative"):
        io.read_table(tmp_path / "neg.csv", tmp_path / "pos.csv")
    _write(tmp_path / "txt.csv", "cell,A,tissue\nc1,1,tumor\nc2,2,stroma\n")
    with pytest.raises(ValueError, match="non-numeric"):
        io.read_table(tmp_path / "txt.csv", tmp_path / "pos.csv")
    a = ad.AnnData(X=np.array([[1, 2], [3, 4]], np.float32))
    b = ad.AnnData(X=np.array([[1, 2], [3, 4]], np.float32))
    a.var_names, b.var_names = ["A", "B"], ["C", "D"]
    for x in (a, b):
        x.obsm["spatial"] = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="share no"):
        pp.concat_slices([a, b], batch_categories=["a", "b"], normalize=False, log1p=False, n_top_genes=1)
    h = ad.AnnData(X=np.ones((3, 2), np.float32))
    h.obs["x"], h.obs["y"] = [0.0, np.nan, 2.0], [0.0, 1.0, 2.0]
    h.write_h5ad(tmp_path / "h.h5ad")
    with pytest.warns(UserWarning, match="without coordinates"):
        r = io.read(tmp_path / "h.h5ad")
    assert r.n_obs == 2 and r.uns["staig"]["qc"]["removed"] == {"missing_coordinates": 1}


def test_file_discovery_respects_pattern_priority_and_ambiguity(tmp_path):
    make_cosmx(tmp_path / "cos", prefix="A_")
    make_cosmx(tmp_path / "cos2", prefix="B_")
    (tmp_path / "cos2" / "B_metadata_file.csv").rename(tmp_path / "cos" / "B_metadata_file.csv")
    (tmp_path / "cos2" / "B_exprMat_file.csv").rename(tmp_path / "cos" / "B_exprMat_file.csv")
    a = io.read_cosmx(tmp_path / "cos", counts=tmp_path / "cos" / "B_exprMat_file.csv", min_counts=0)
    assert a.uns["staig"]["source"].endswith("B_exprMat_file.csv")
    assert io._first(tmp_path / "cos", "B_metadata_file.csv", "*metadata_file.csv").name == "B_metadata_file.csv"
    make_xenium(tmp_path / "amb")
    make_merscope(tmp_path / "amb" / "mer")
    with pytest.raises(ValueError, match="several layouts"):
        io.detect_platform(tmp_path / "amb")
    make_seqfish(tmp_path / "sf")
    (tmp_path / "sf" / "Roi1_CellxGene.csv").rename(tmp_path / "sf" / "Roi2_CellxGene.csv")
    make_seqfish(tmp_path / "sf2")
    (tmp_path / "sf2" / "Roi1_CellxGene.csv").rename(tmp_path / "sf" / "Roi1_CellxGene.csv")
    with pytest.raises(FileNotFoundError, match="several CellxGene"):
        io.read_seqfish(tmp_path / "sf")


def test_slide_seq_orientation_with_partial_locations(tmp_path):
    _write(tmp_path / "Puck.digital_expression.txt", "GENE\tc1\tc2\tc3\tc4\nA\t1\t2\t3\t4\nB\t5\t6\t7\t8\n")
    _write(tmp_path / "Puck_bead_locations.csv", "barcodes,xcoord,ycoord\nc1,1,2\n")
    a = io.read(tmp_path, min_counts=0)
    assert list(a.obs_names) == ["c1"] and a.X.toarray().tolist() == [[1.0, 5.0]]
    assert a.uns["staig"]["qc"]["removed"]["no_coordinates"] == 3


def test_large_tables_are_converted_in_chunks(tmp_path):
    df = pd.DataFrame(rng.poisson(1, (50, 7)), index=[f"c{i}" for i in range(50)], columns=list("ABCDEFG"))
    a = io._counts_frame_to_anndata(df, chunk=8)
    assert sp.isspmatrix_csr(a.X) and np.array_equal(a.X.toarray(), df.to_numpy())


def test_stereo_bins_on_absolute_grid_and_header_kept(tmp_path):
    make_gem(tmp_path / "a.gem")
    a = io.read(tmp_path / "a.gem", bin_size=4, min_counts=0)
    assert a.uns["staig"]["gem_header"]["OffsetX"] == "100" and a.uns["staig"]["gem_header"]["FileFormat"] == "GEMv0.1"
    assert np.allclose(a.obsm["spatial"].min(0), [102, 202])                        # bin 25 of width 4 -> centre 102
    assert all(n.startswith("bin4_") for n in a.obs_names)


def test_transcript_positions_are_count_weighted(tmp_path):
    _write(tmp_path / "w.csv", "cell_id,gene,x,y,count\n1,A,0,0,9\n1,A,10,0,1\n")
    a = io.read_transcripts(tmp_path / "w.csv", min_counts=0)
    assert a.X.toarray().tolist() == [[10.0]] and np.allclose(a.obsm["spatial"], [[1.0, 0.0]])
