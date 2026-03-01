import numpy as np
from src.evaluation import plots
from pathlib import Path


def test_reliability_diagram_and_plot(tmp_path):
    conf = np.array([0.1, 0.2, 0.7, 0.9, 0.85])
    labels = np.array([0, 0, 1, 1, 1])
    bc, acc = plots.reliability_diagram(conf, labels, num_bins=5)
    assert len(bc) == 5
    out = tmp_path / "rel.png"
    plots.plot_reliability_diagram(conf, labels, str(out), num_bins=5)
    assert out.exists()


def test_confusion_matrix_plot(tmp_path):
    cm = np.array([[5, 2], [1, 7]])
    labels = ["ENTAIL", "CONTRADICT"]
    out = tmp_path / "cm.png"
    plots.plot_confusion_matrix(cm, labels, str(out))
    assert out.exists()
