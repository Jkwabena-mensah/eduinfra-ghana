"""
apply_ux_fixes.py  (v2 — targeted patches only)
================================================
Applies the 6 remaining UX fixes to app.py.
Run once from project root:
    python apply_ux_fixes.py
    streamlit run app.py
"""
import py_compile, shutil
from pathlib import Path

APP = Path(__file__).parent / "app.py"
BAK = Path(__file__).parent / "app.py.bak"

with open(APP, encoding="utf-8") as f:
    app = f.read()

# Guard: if app.py is corrupted (placeholder), stop
if len(app.splitlines()) < 100:
    print("❌ app.py appears corrupted (too short).")
    print("   Run: git checkout app.py   to restore from git.")
    exit(1)

shutil.copy(APP, BAK)
changes = 0

# ── 1. MutationObserver JS ────────────────────────────────────────────────
if "tryAttach(20)" in app or ("tryAttach" in app and "MutationObserver" not in app):
    js_start = app.find('    progressive_js = f"""')
    js_end   = app.find('    m.get_root().html.add_child(Element(progressive_js))')
    js_end  += len('    m.get_root().html.add_child(Element(progressive_js))')
    if js_start > 0 and js_end > js_start:
        new_js = '''    progressive_js = f"""
    <script>
    (function() {{
        function applyZoomVisibility(map) {{
            var threshold = {ZOOM_THRESHOLD};
            var zoom = map.getZoom();
            map.eachLayer(function(layer) {{
                if (layer instanceof L.MarkerClusterGroup) {{
                    if (zoom >= threshold) {{
                        if (!map.hasLayer(layer)) map.addLayer(layer);
                    }} else {{
                        if (map.hasLayer(layer)) map.removeLayer(layer);
                    }}
                }}
            }});
        }}
        function attachToMap() {{
            var maps = [];
            try {{
                maps = Object.values(window).filter(function(v) {{
                    return v && typeof v.getZoom === \'function\' && typeof v.eachLayer === \'function\';
                }});
            }} catch(e) {{}}
            if (!maps.length) return false;
            var map = maps[0];
            applyZoomVisibility(map);
            var _pending = false;
            map.on(\'zoomend\', function() {{
                if (_pending) return;
                _pending = true;
                setTimeout(function() {{ applyZoomVisibility(map); _pending = false; }}, 50);
            }});
            map.on(\'moveend\', function() {{ applyZoomVisibility(map); }});
            return true;
        }}
        if (!attachToMap()) {{
            var obs = new MutationObserver(function(_, o) {{
                if (attachToMap()) o.disconnect();
            }});
            obs.observe(document.body, {{ childList: true, subtree: true }});
            setTimeout(function() {{ obs.disconnect(); }}, 10000);
        }}
    }})();
    </script>
    """
    m.get_root().html.add_child(Element(progressive_js))'''
        app = app[:js_start] + new_js + app[js_end:]
        changes += 1
        print("✅ MutationObserver JS — polling loop replaced")
    else:
        print("⚠️  JS block markers not found")
else:
    print("⏭  MutationObserver already applied — skipping")

# ── 2. Tab 1 map height 600 → 640 ─────────────────────────────────────────
old_h = 'st_folium(m, width="100%", height=600, returned_objects=[])'
new_h = 'st_folium(m, width="100%", height=640, returned_objects=[])'
if old_h in app:
    app = app.replace(old_h, new_h)
    changes += 1
    print("✅ Tab 1 map height: 600 → 640px")
else:
    print("⏭  Tab 1 height already 640 — skipping")

# ── 3. CSS: iframe + button hover + chat input ────────────────────────────
CSS_ANCHOR = "    </style>\n    \"\"\","
EXTRA_CSS = """
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
    }
"""
if CSS_ANCHOR in app and 'streamlit_folium.st_folium' not in app:
    app = app.replace(CSS_ANCHOR, EXTRA_CSS + CSS_ANCHOR)
    changes += 1
    print("✅ CSS: iframe, button hover, chat input, multiselect tags")
elif 'streamlit_folium.st_folium' in app:
    print("⏭  CSS already applied — skipping")
else:
    print("⚠️  CSS anchor not found")

# ── Write & verify ────────────────────────────────────────────────────────
with open(APP, "w", encoding="utf-8") as f:
    f.write(app)

try:
    py_compile.compile(str(APP), doraise=True)
    print(f"\n✅ Done — {changes} fixes applied | {len(app.splitlines())} lines | syntax OK")
    print("   Run: streamlit run app.py")
except py_compile.PyCompileError as e:
    print(f"\n❌ Syntax error — restoring backup: {e}")
    shutil.copy(BAK, APP)
