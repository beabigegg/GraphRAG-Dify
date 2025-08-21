SECTION_RE = r"^(\d+(?:\.\d+){0,3})\s+"
CODE_RE = r"[A-Z]{1,3}-SP\d{3}|DB-PA\d{3}|DB-OC\d{3}|F-[A-Z]{2}\d{4}[A-Z0-9]?"
NUMUNIT_RE = r"(≤|≥|=|±)?\s*\d+(?:\.\d+)?\s*(um|µm|g|°C|%|mil|hrs?|hours?)"
FREQ_WORDS = [
    "每班","每兩小時","每天","開機","維修","換Type","Every shift","Every two hours","Daily"
]