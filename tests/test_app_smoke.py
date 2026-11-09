"""Streamlit is not installable in every environment, so this drives every page function against a stub
streamlit module. It proves the page code runs on the real run outputs; it does NOT test the visual layout."""
import sys
import types
import unittest
from pathlib import Path
from unittest import mock
import helpers  # noqa


class State(dict):
    __getattr__ = dict.get


def make_stub():
    st = mock.MagicMock(name="streamlit")
    st.session_state = State()
    st.columns.side_effect = lambda n, *a, **k: [st for _ in range(n if isinstance(n, int) else len(n))]  # columns delegate to st
    st.multiselect.return_value = []
    st.slider.return_value = 0.0
    st.checkbox.return_value = True
    st.button.return_value = False
    st.file_uploader.return_value = []
    st.selectbox.side_effect = lambda label, options, *a, **k: options[0]
    st.sidebar.selectbox.side_effect = lambda label, options, *a, **k: options[0]
    return st


class TestAppSmoke(unittest.TestCase):
    def test_all_pages_render_on_real_outputs(self):
        run_dir = Path(__file__).resolve().parents[1] / "outputs" / "run_heuristic"
        if not (run_dir / "run_manifest.json").exists():
            self.skipTest("no run outputs yet")
        stub = make_stub()
        pandas_ok = True
        try:
            import pandas  # noqa
        except ImportError:
            pandas_ok = False
        if not pandas_ok:
            self.skipTest("pandas missing")
        with mock.patch.dict(sys.modules, {"streamlit": stub}):
            sys.modules.pop("app", None)
            import app
            run = app.load_run("run_heuristic")
            self.assertTrue(run["ranked"])
            app.page_dashboard(run, ["run_heuristic"])
            app.page_ranked(run)
            app.page_details(run)
            app.page_quality(run)
            app.page_export(run)
            self.assertTrue(stub.download_button.called)
            self.assertTrue(stub.metric.called)


if __name__ == "__main__":
    unittest.main()
