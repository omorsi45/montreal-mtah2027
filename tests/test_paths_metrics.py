import numpy as np

from montreal import paths
from montreal.metrics import score

H, A, D, X = 0, 1, 2, 3


def test_upstream_importable():
    games, pm = paths.upstream()
    assert games.half_video_blocks("BAR-ATH") == [(1, 11185, 90651), (2, 110808, 190197)]
    assert pm.H == 0 and pm.X == 3


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("MTAH_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("MONTREAL_SAMPLE_DIR", str(tmp_path / "s"))
    assert paths.data_dir() == tmp_path / "d"
    assert paths.sample_dir() == tmp_path / "s"


def test_default_paths(monkeypatch):
    for k in ("MTAH_DATA_DIR", "MONTREAL_SAMPLE_DIR", "MTAH_REPO_DIR"):
        monkeypatch.delenv(k, raising=False)
    assert paths.sample_dir() == paths.repo_dir() / "external" / "metrica-sample-data"
    assert paths.data_dir() == paths.upstream_dir() / "data"
    assert paths.upstream_dir() == paths.repo_dir().parent / "more-than-a-hack-2027"


def test_cache_dir_created():
    c = paths.cache_dir()
    assert c.is_dir() and c.name == ".cache"


def _gt():
    return np.array([X] * 10 + [H] * 100 + [A] * 100 + [D] * 50 + [X] * 10, dtype=np.int8)


def test_score_perfect_and_constant():
    gt = _gt()
    perfect = score(gt, gt)
    assert perfect["spell_f1"] == 1.0
    assert perfect["macro_f1"] == 1.0
    assert perfect["n_gt_spells"] == 2
    assert perfect["n_pred_spells"] == 2
    const = score(gt, np.full(len(gt), D, dtype=np.int8))
    assert const["spell_f1"] == 0.0
    assert const["n_pred_spells"] == 0
    assert const["n_gt_spells"] == 2
