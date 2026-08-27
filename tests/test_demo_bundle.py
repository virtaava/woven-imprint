from pathlib import Path


def test_built_bundle_references_new_endpoints():
    static = Path(__file__).resolve().parent.parent / "src" / "woven_imprint" / "demo_static" / "assets"
    js = "".join(p.read_text(encoding="utf-8", errors="ignore") for p in static.glob("*.js"))
    for needle in ("/api/memory/pinned", "/api/facts/", "Things the character always remembers"):
        assert needle in js, f"bundle stale: {needle} missing — run `cd demo && npm run build`"
