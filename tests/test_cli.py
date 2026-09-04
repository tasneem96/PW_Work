"""The CLI is the interface the next phases build on; keep it working."""

from __future__ import annotations

import pytest

from braid.cli import main


def test_protocol_validate_and_hash(capsys):
    assert main(["protocol", "validate"]) == 0
    out = capsys.readouterr().out
    assert "valid, status=frozen" in out
    assert main(["protocol", "hash"]) == 0
    assert len(capsys.readouterr().out.strip()) == 64


def test_phase0_gate_command(capsys):
    assert main(["phase0", "gate"]) == 0
    assert "Phase 0 exit gate: PASS" in capsys.readouterr().out


def test_trace_command_prints_one_full_trace(capsys):
    code = main(
        [
            "phase1",
            "trace",
            "--dataset",
            "syn-clusters-d64",
            "--n",
            "400",
            "--M",
            "8",
            "--ef-search",
            "20",
            "--events",
            "5",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "L(q) size:" in out and "exposed local edges:" in out
    assert "neighbor_list_exposed" in out or "entry_point" in out


def test_sweep_and_gate_commands(tmp_path, capsys):
    code = main(
        [
            "phase1",
            "sweep",
            "--profile",
            "smoke",
            "--out",
            str(tmp_path / "run"),
            "--n",
            "400",
            "--queries",
            "16",
            "--trace-sample",
            "3",
            "--no-parity",
            "--quiet",
        ]
    )
    assert code == 0
    assert "cells: 6 / 6" in capsys.readouterr().out
    # parity was skipped, so the parity check is unavailable but only advisory
    code = main(["phase1", "gate", "--summary", str(tmp_path / "run" / "summary.json")])
    out = capsys.readouterr().out
    assert code == 0
    assert "Phase 1 exit gate: PASS" in out
    assert "[warn] native_parity_available" in out


def test_unknown_profile_is_reported(tmp_path):
    from braid.protocol import ProtocolError

    with pytest.raises(ProtocolError):
        main(["phase1", "sweep", "--profile", "nope", "--out", str(tmp_path)])
