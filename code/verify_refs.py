"""Verify every reference of the submitted VTSE manuscript against Crossref (editor point 12).
For each \\bibitem: the DOI must resolve, the Crossref title must match the cited title (similarity >= 0.85
after normalisation) and the year must agree (+-1). References without a DOI are searched by title.
Writes results/ref_check.json and prints problems."""
import difflib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE.parents[1] / "MEARE_IA_Resubmission" / "vtse" / "main.tex"
UA = {"User-Agent": "ref-verifier/1.0 (mailto:victor-csme@dsu.edu.in)"}


def norm(s):
    s = re.sub(r"<[^>]+>|\\[a-zA-Z]+|[{}]", " ", s.lower())
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", s)).strip()


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return json.loads(r.read().decode("utf8"))
        except Exception as e:  # noqa: BLE001
            err = e; time.sleep(2 * (k + 1))
    raise err


def main():
    tex = TEX.read_text(encoding="utf8")
    bib = tex[tex.index("\\begin{thebibliography}"):]
    items = re.findall(r"\\bibitem\{b(\d+)\}\s*(.*?)(?=\\bibitem|\\end\{thebibliography\})", bib, re.S)
    out = []
    for n, txt in items:
        txt = " ".join(txt.split())
        title = (re.search(r"\u201c(.+?),?\u201d", txt) or re.search(r"``(.+?),?''", txt)
                 or re.search(r"\"(.+?),?\"", txt))
        title = title.group(1).strip(" ,") if title else ""
        first_author = re.split(r",| and ", txt)[0].split()[-1] if txt else ""
        year = re.findall(r"\b(19\d\d|20\d\d)\b", txt)
        year = int(year[-1]) if year else None
        doi = re.search(r"doi\.org/([^}]+)\}", txt) or re.search(r"doi: (10\.\S+?)\.?$", txt)
        doi = doi.group(1).replace("\\_", "_") if doi else None
        rec = dict(n=int(n), title=title, year=year, doi=doi, first_author=first_author)
        try:
            if doi:
                m = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))["message"]
            else:
                m = get("https://api.crossref.org/works?rows=1&query.bibliographic=" + urllib.parse.quote(title))["message"]["items"][0]
            ct = (m.get("title") or [""])[0]
            cy = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
            sim = difflib.SequenceMatcher(None, norm(ct), norm(title)).ratio()
            rec.update(crossref_title=ct, crossref_year=cy, crossref_doi=m.get("DOI"), sim=round(sim, 3),
                       container=(m.get("container-title") or [""])[0],
                       authors=[(a.get("given", ""), a.get("family", "")) for a in m.get("author", [])[:6]],
                       author_ok=bool(not m.get("author") or norm(first_author) == norm(m["author"][0].get("family", "")) or norm(first_author) in norm(" ".join(a.get("family", "") for a in m["author"]))),
                       ok=bool(sim >= 0.85 and (year is None or cy is None or abs(cy - year) <= 1)))
        except Exception as e:  # noqa: BLE001
            rec.update(ok=False, error=str(e)[:120])
        out.append(rec)
        if not rec.get("ok") or not rec.get("author_ok", True):
            print("PROBLEM", n, rec.get("sim"), rec.get("year"), rec.get("crossref_year"), "|", title[:70], "|",
                  rec.get("crossref_title", rec.get("error", ""))[:70], "| author:", first_author, rec.get("authors", [""])[:1], flush=True)
        time.sleep(0.25)
    (HERE / "results" / "ref_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf8")
    print("checked", len(out), "ok", sum(r.get("ok", False) for r in out))


if __name__ == "__main__":
    main()
