from src.utils.normalization import canonical, normalize_codes

def test_canonical():
    assert canonical("  A  B  ") == "a b"

def test_code_extract():
    s = "見 DB-SP015 與 F-RD09M4"
    codes = normalize_codes(s)
    assert "DB-SP015" in codes and "F-RD09M4" in codes