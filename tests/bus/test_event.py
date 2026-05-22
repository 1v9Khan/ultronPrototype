"""Tests for ultron.bus.event primitives."""

from __future__ import annotations

from ultron.bus.event import BusEvent, EventPayload


# ---------------------------------------------------------------------------
# BusEvent.define
# ---------------------------------------------------------------------------


def test_define_returns_frozen_dataclass() -> None:
    e = BusEvent.define("test.event", {"x": int})
    assert e.type == "test.event"
    assert e.schema == {"x": int}
    assert e.description == ""


def test_define_with_description() -> None:
    e = BusEvent.define("test.event", {"x": int}, description="hello")
    assert e.description == "hello"


def test_define_schema_is_copied() -> None:
    source = {"x": int}
    e = BusEvent.define("test.event", source)
    source["y"] = str
    # Mutating the source dict after define must not change the stored schema.
    assert e.schema == {"x": int}


# ---------------------------------------------------------------------------
# BusEvent.validate
# ---------------------------------------------------------------------------


def test_validate_ok_on_match() -> None:
    e = BusEvent.define("test.event", {"x": int, "y": str})
    assert e.validate({"x": 1, "y": "hello"}) is None


def test_validate_missing_field() -> None:
    e = BusEvent.define("test.event", {"x": int, "y": str})
    problem = e.validate({"x": 1})
    assert problem is not None
    assert "missing" in problem.lower()
    assert "y" in problem


def test_validate_wrong_type() -> None:
    e = BusEvent.define("test.event", {"x": int})
    problem = e.validate({"x": "not an int"})
    assert problem is not None
    assert "int" in problem


def test_validate_none_value_passes() -> None:
    # None is treated as "absent" -- a field with None passes type check,
    # caller decides whether None is acceptable (Optional semantics).
    e = BusEvent.define("test.event", {"x": int})
    assert e.validate({"x": None}) is None


def test_validate_empty_schema_accepts_anything() -> None:
    e = BusEvent.define("test.event", {})
    assert e.validate({}) is None
    assert e.validate({"random": "stuff"}) is None


def test_validate_multiple_problems_combined() -> None:
    e = BusEvent.define("test.event", {"x": int, "y": str})
    problem = e.validate({"x": "bad", "y": 5})
    assert problem is not None
    assert "x" in problem
    assert "y" in problem


# ---------------------------------------------------------------------------
# EventPayload.make
# ---------------------------------------------------------------------------


def test_make_generates_id_when_omitted() -> None:
    e = BusEvent.define("test.event", {"x": int})
    p = EventPayload.make(e, {"x": 1})
    assert p.id.startswith("evt_")
    assert p.type == "test.event"
    assert p.properties == {"x": 1}
    assert p.published_at > 0


def test_make_respects_explicit_id() -> None:
    e = BusEvent.define("test.event", {"x": int})
    p = EventPayload.make(e, {"x": 1}, id="my-custom-id")
    assert p.id == "my-custom-id"


def test_make_copies_properties() -> None:
    e = BusEvent.define("test.event", {"x": int})
    props = {"x": 1}
    p = EventPayload.make(e, props)
    props["y"] = 2  # Mutate source after make.
    assert "y" not in p.properties


def test_make_generates_unique_ids() -> None:
    e = BusEvent.define("test.event", {"x": int})
    ids = {EventPayload.make(e, {"x": i}).id for i in range(50)}
    assert len(ids) == 50
