"""
fix_css.py — fixes the broken CSS block that causes NameError: 'border'
Run once:  .venv\Scripts\python.exe fix_css.py
"""
import shutil
from pathlib import Path

APP = Path(__file__).parent / "app.py"
BAK = Path(__file__).parent / "app.py.css_bak"

with open(APP, encoding="utf-8") as f:
    src = f.read()

# The broken block: CSS sitting OUTSIDE the f-string (single braces, not double)
# We need to:
# 1. Remove the broken CSS block that's outside the f-string
# 2. Insert correctly escaped CSS (double braces) INSIDE the f-string before </style>

BROKEN_OUTSIDE = """
    /* ── Folium iframe UX ── */
    iframe[title="streamlit_folium.st_folium"] {
        border: none !important;
        border-radius: 10px;
        display: block;
    }
    /* ── Button hover lift ── */
    .stButton > button { transition: all 0.18s ease !important; }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 16px rgba(252,209,22,0.25) !important;
    }
    /* ── Chat input polish ── */
    [data-testid="stChatInput"] textarea {
        border-radius: 20px !important;
        border: 1px solid rgba(252,209,22,0.3) !important;
        background: rgba(22,27,34,0.8) !important;
        color: #E6EDF3 !important;
        font-size: 0.88rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: rgba(252,209,22,0.7) !important;
        box-shadow: 0 0 0 2px rgba(252,209,22,0.15) !important;
    }
    /* ── Multiselect tag gold ── */
    [data-baseweb="tag"] {
        background: rgba(252,209,22,0.15) !important;
        border: 1px solid rgba(252,209,22,0.4) !important;
        color: #FCD116 !important;
        font-weight: 600 !important;
        border-radius: 20px !important;
    }"""

CORRECT_INSIDE = """
    /* ── Folium iframe UX ── */
    iframe[title="streamlit_folium.st_folium"] {{
        border: none !important;
        border-radius: 10px;
        display: block;
    }}
    /* ── Button hover lift ── */
    .stButton > button {{ transition: all 0.18s ease !important; }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 16px rgba(252,209,22,0.25) !important;
    }}
    /* ── Chat input polish ── */
    [data-testid="stChatInput"] textarea {{
        border-radius: 20px !important;
        border: 1px solid rgba(252,209,22,0.3) !important;
        background: rgba(22,27,34,0.8) !important;
        color: #E6EDF3 !important;
        font-size: 0.88rem !important;
    }}
    [data-testid="stChatInput"] textarea:focus {{
        border-color: rgba(252,209,22,0.7) !important;
        box-shadow: 0 0 0 2px rgba(252,209,22,0.15) !important;
    }}
    /* ── Multiselect tag gold ── */
    [data-baseweb="tag"] {{
        background: rgba(252,209,22,0.15) !important;
        border: 1px solid rgba(252,209,22,0.4) !important;
        color: #FCD116 !important;
        font-weight: 600 !important;
        border-radius: 20px !important;
    }}"""

# Step 1: check broken block exists
if BROKEN_OUTSIDE not in src:
    print("❌ Broken CSS block not found — may already be fixed or has different formatting.")
    print("   Checking if app runs cleanly...")
    import py_compile
    try:
        py_compile.compile(str(APP), doraise=True)
        print("✅ app.py syntax is OK — no fix needed.")
    except py_compile.PyCompileError as e:
        print(f"❌ Syntax error still present: {e}")
        print("   Please share the error line number for manual fix.")
    exit()

shutil.copy(APP, BAK)
print(f"Backup: {BAK}")

# Step 2: Remove broken outside CSS, insert correct inside CSS before </style>
src = src.replace(BROKEN_OUTSIDE, "")  # remove broken block

# Step 3: Insert correct CSS inside the f-string, before </style>
STYLE_CLOSE = "    </style>\n    \"\"\","
if STYLE_CLOSE not in src:
    print("❌ Could not find </style> anchor.")
    exit()

src = src.replace(STYLE_CLOSE, CORRECT_INSIDE + "\n    </style>\n    \"\"\",", 1)

# Step 4: Write and verify
with open(APP, "w", encoding="utf-8") as f:
    f.write(src)

import py_compile
try:
    py_compile.compile(str(APP), doraise=True)
    print(f"✅ CSS fixed — {len(src.splitlines())} lines, syntax OK")
    print("   Run: .venv\\Scripts\\streamlit run app.py")
except py_compile.PyCompileError as e:
    print(f"❌ Syntax error after fix: {e}")
    print("   Restoring backup...")
    shutil.copy(BAK, APP)
    print("   Backup restored.")
