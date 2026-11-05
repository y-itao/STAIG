import os
import tempfile

import pytest

from staig.gui import run_pipeline


def test_gui_backend_runs_on_synthetic_h5ad():
    from test_pipeline import make_synthetic

    a = make_synthetic(seed=2)
    d = tempfile.mkdtemp()
    p = os.path.join(d, "syn.h5ad")
    a.write_h5ad(p)
    met, fig, h5, log = run_pipeline(p, "", 4, "auto", "", 1, False, 15, True, d)
    assert '"ARI"' in met and os.path.exists(h5) and "written" in log
    assert fig is None or os.path.exists(fig)


def test_gui_app_builds_if_gradio_installed():
    pytest.importorskip("gradio")
    from staig.gui import build_app

    app = build_app()
    assert app is not None
