"""Render HANDOFF.md -> PDF with rendered mermaid diagram (HTML + headless Chromium)."""
import re, sys, pathlib, markdown

src = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2])
text = src.read_text()

# pull out mermaid fenced blocks, render the rest as markdown
mer = []
def stash(m):
    mer.append(m.group(1)); return f"@@MERMAID{len(mer)-1}@@"
text = re.sub(r"```mermaid\n(.*?)```", stash, text, flags=re.S)
html_body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
for i, code in enumerate(mer):
    html_body = html_body.replace(f"<p>@@MERMAID{i}@@</p>", f'<pre class="mermaid">{code}</pre>')

html = f"""<!doctype html><html><head><meta charset="utf-8">
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:24px auto;color:#1a1a1a;line-height:1.5;padding:0 20px}}
 h1{{border-bottom:3px solid #1565c0;padding-bottom:6px}} h2{{border-bottom:1px solid #ddd;margin-top:28px}}
 code{{background:#f3f3f3;padding:1px 4px;border-radius:3px;font-size:90%}}
 pre{{background:#1115;padding:10px;border-radius:6px;overflow:auto}} pre code{{background:none}}
 table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ccc;padding:5px 8px;font-size:90%}} th{{background:#eef}}
 blockquote{{border-left:4px solid #1565c0;margin:0;padding:4px 12px;background:#f7faff;color:#333}}
 .mermaid{{background:#fff;text-align:center}}
</style>
<script type="module">
 import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
 mermaid.initialize({{startOnLoad:true}});
 window.__merdone=false; await mermaid.run(); window.__merdone=true;
</script></head><body>{html_body}</body></html>"""
htmlfile = out.with_suffix(".html"); htmlfile.write_text(html)

from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    pg.goto(f"file://{htmlfile.resolve()}", wait_until="networkidle")
    try:
        pg.wait_for_function("window.__merdone===true", timeout=15000)
    except Exception:
        pass
    pg.wait_for_timeout(1500)
    pg.pdf(path=str(out), format="A4", margin={"top":"14mm","bottom":"14mm","left":"12mm","right":"12mm"},
           print_background=True)
    b.close()
print("PDF ->", out)
