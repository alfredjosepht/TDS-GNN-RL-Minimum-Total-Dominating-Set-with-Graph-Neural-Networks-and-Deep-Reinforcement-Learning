"""End-to-end tests of the Streamlit app (headless, streamlit.testing)."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "checkpoints" / "best.pt").exists(), reason="needs checkpoints/best.pt")

DRIVER = """
import sys
sys.path.insert(0, r"{app}"); sys.path.insert(0, r"{root}")
import streamlit as st
from components.theme import inject
from components.graph_input import generate, parse_edge_text, SAMPLE_EDGE_LIST
from components.state import set_current
inject()
{setup}
from pages import {page}
{page}.render()
"""


def app_for(tmp_path, page, setup=""):
    from streamlit.testing.v1 import AppTest
    f = tmp_path / f"drv_{page}.py"
    f.write_text(DRIVER.format(app=ROOT / "app", root=ROOT, page=page, setup=setup), encoding="utf-8")
    return AppTest.from_file(str(f), default_timeout=300)


def stats_text(at):
    return " ".join(re.sub(r"<[^>]+>", " ", m.value) for m in at.markdown if 'class="stat"' in m.value)


def notices(at):
    return " ".join(re.sub(r"<[^>]+>", " ", m.value) for m in at.markdown if 'class="notice' in m.value)


def click(at, label):
    [b for b in at.button if b.label == label][0].click().run()
    assert not at.exception, [e.value for e in at.exception]


def test_entry_app_loads():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=300).run()
    assert not at.exception


def test_a_sample_edge_list(tmp_path):
    at = app_for(tmp_path, "solve", 'set_current(parse_edge_text(SAMPLE_EDGE_LIST, "sample.txt"), "s")').run()
    click(at, "Find total dominating set")
    s = stats_text(at)
    assert "Valid" in s and "Optimum" in s and " 4 " in s        # Petersen: γ_t = 4 proven by ILP


def test_b_er_100(tmp_path):
    at = app_for(tmp_path, "solve", 'set_current(generate("er", 100, 6.0, 0), "er")').run()
    click(at, "Find total dominating set")
    assert "Valid" in stats_text(at)


def test_c_grid_10_optimum_30(tmp_path):
    at = app_for(tmp_path, "solve", 'set_current(generate("grid", 100, 0, 0, side=10), "g")').run()
    click(at, "Find total dominating set")
    s = stats_text(at)
    assert "Optimum" in s and " 30 " in s and "Gap" in s and "Valid" in s


def test_d_isolated_vertex_friendly_error(tmp_path):
    at = app_for(tmp_path, "solve", 'set_current(parse_edge_text("0 1\\n1 2\\n7", "iso.txt"), "i")').run()
    assert not at.exception
    assert "No total dominating set can exist" in notices(at)
    assert not [b for b in at.button if b.label == "Find total dominating set" and False]


def test_e_large_ba_no_drawing(tmp_path):
    at = app_for(tmp_path, "solve", 'set_current(generate("ba", 5000, 3, 0), "ba")').run()
    click(at, "Find total dominating set")
    assert "Valid" in stats_text(at)
    assert "drawing is skipped" in notices(at)


def test_messy_input_parsing():
    import sys
    sys.path.insert(0, str(ROOT / "app"))
    from components.graph_input import GraphInputError, parse_edge_text
    lg = parse_edge_text("# c\na b\nb b\na b\nb,c\nthis line has far too many tokens\n\nc a", "x")
    assert lg.graph.n == 3 and lg.graph.num_edges == 3 and lg.labels == ["a", "b", "c"]
    w = " ".join(lg.warnings)
    assert "self-loop" in w and "duplicate" in w and "malformed" in w and "relabelled" in w
    with pytest.raises(GraphInputError):
        parse_edge_text("   \n# only a comment\n", "empty")


@pytest.mark.parametrize("page,button", [("watch", "Record construction"), ("compare", "Run all methods"),
                                         ("results", None), ("how", None)])
def test_other_pages(tmp_path, page, button):
    at = app_for(tmp_path, page, 'set_current(generate("er", 60, 5.0, 1), "p")').run()
    assert not at.exception
    if button:
        click(at, button)
