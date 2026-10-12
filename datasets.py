"""Helpers to build structure + potency datasets for virtual-site benchmarking.

Every dataset is described by one ``SOURCES`` entry in the notebook: a source of crystal
structures and a source of potency/affinity measurements. Only compounds present in both are
kept; compounds are matched on the first two InChIKey blocks (connectivity + stereo, ignoring
protonation state), or on connectivity alone when the measured sample's stereo was unassigned::

    "<display name>": {
        "structures": {"from": "<kind>", ...},   # kinds: zip, pdb_group, polaris
        "potency": {"from": "<kind>", ...},      # kinds: csv, chembl, polaris
        "meta": {"target": "..."},               # optional constant columns
    }

See the ``structures_*`` / ``potency_*`` functions below for the options of each kind.

``load_all`` builds the table of structure-potency pairs (``data/<name>/pairs.csv``).
``write_structures`` additionally writes the crystal ligand and the protein of every pair to
``data/<name>/structures/<structure_id>/{ligand.sdf,protein.pdb}``. Raw downloads are cached in
``data/raw/``.
"""

import fnmatch
import hashlib
import io
import json
import pickle
import re
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
RT = 0.0019872 * 298.15  # kcal/mol at 25 C
UNIT_TO_M = {"M": 1, "mM": 1e-3, "uM": 1e-6, "µM": 1e-6, "nM": 1e-9}
PAIR_COLUMNS = ["dataset", "structure_id", "compound_id", "smiles", "potency_type", "pX",
                "n_measurements", "expt_dg", "potency_stereo"]
INTERNAL_COLUMNS = ["mol", "protein_ref", "key", "skeleton"]


# --- downloading --------------------------------------------------------------------------


def _retry(fn, retries=6):
    for i in range(retries):
        try:
            return fn()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(5 * (i + 1))


def http_get(url, data=None):
    """GET (or POST ``data`` as JSON) ``url`` with retries; responses are cached in ``data/raw/http``."""
    cache = RAW_DIR / "http" / hashlib.sha1((url + json.dumps(data)).encode()).hexdigest()
    if not cache.exists():
        headers = {"User-Agent": "vsites-datasets", "Content-Type": "application/json"}
        req = urllib.request.Request(url, data=json.dumps(data).encode() if data else None, headers=headers)
        body = _retry(lambda: urllib.request.urlopen(req, timeout=120).read())
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(body)
    return cache.read_bytes()


def download(url):
    """Download a (large) file once into ``data/raw/`` and return its path."""
    name = Path(url.split("?")[0]).name
    dest = RAW_DIR / f"{hashlib.sha1(url.encode()).hexdigest()[:8]}_{name}"
    if not dest.exists():
        print(f"Downloading {url}")
        dest.parent.mkdir(parents=True, exist_ok=True)

        def fetch():
            req = urllib.request.Request(url, headers={"User-Agent": "vsites-datasets"})
            with urllib.request.urlopen(req, timeout=120) as r, open(f"{dest}.part", "wb") as f:
                shutil.copyfileobj(r, f)

        _retry(fetch)
        Path(f"{dest}.part").rename(dest)
    return dest


def raw_github_url(repo_url, path):
    """Turn a ``github.com/<owner>/<repo>[/tree/<ref>]`` URL plus a file path into a raw download URL."""
    m = re.match(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/tree/([^/]+))?/?$", repo_url)
    if not m:
        return f"{repo_url.rstrip('/')}/{path}"
    owner, repo, ref = m.groups()
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref or 'main'}/{path}"


def is_zip(url):
    return Path(url.split("?")[0]).suffix == ".zip"


def zip_member(z, pattern):
    """Contents of the single member of zip archive ``z`` matching the glob ``pattern``."""
    matches = fnmatch.filter(z.namelist(), pattern)
    if len(matches) != 1:
        raise ValueError(f"{pattern!r} matches {len(matches)} files in {z.filename}")
    return z.read(matches[0])


def read_csv(url, file):
    """CSV ``file`` from a GitHub repo / base URL, or from inside a zip archive at ``url``."""
    if is_zip(url):
        return pd.read_csv(io.BytesIO(zip_member(zipfile.ZipFile(download(url)), file)))
    return pd.read_csv(download(raw_github_url(url, file)))


def cached(fn, **params):
    """Run a (slow, networked) fetcher once and cache its DataFrame in ``data/raw/``."""
    key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    path = RAW_DIR / f"{fn.__name__}-{key}.pkl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(fn(**params), open(path, "wb"))
    return pickle.load(open(path, "rb"))


# --- chemistry / structure helpers ----------------------------------------------------------


def match_key(mol):
    """Compound identity used to pair structures with potencies: InChIKey without the protonation flag."""
    if mol is None:
        return None
    key = Chem.MolToInchiKey(mol)
    return key[:25] if key else None


def stereo_label(mol):
    """Stereo of the measured sample: 'racemic' (AND group), 'relative' (OR group),
    'unspecified' (stereocentres left unassigned) or 'defined'."""
    kinds = {str(g.GetGroupType()) for g in mol.GetStereoGroups()}
    if any("AND" in k for k in kinds):
        return "racemic"
    if any("OR" in k for k in kinds):
        return "relative"
    centres = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    if any(label == "?" for _, label in centres):
        return "unspecified"
    return "defined"


def to_pdb(atoms):
    """biotite AtomArray -> PDB text."""
    from biotite.structure.io.pdb import PDBFile

    f, buf = PDBFile(), io.StringIO()
    f.set_structure(atoms)
    f.write(buf)
    return buf.getvalue()


def clean_protein(text, fmt="pdb"):
    """Protein from a deposited PDB/mmCIF: amino acids and ions only (no ligands, waters or
    additives), keeping the highest-occupancy alternate conformation; returned as PDB text."""
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile
    from biotite.structure.io.pdbx import CIFFile, get_structure

    kw = dict(model=1, altloc="occupancy", extra_fields=["occupancy", "b_factor"])
    if fmt == "cif":
        atoms = get_structure(CIFFile.read(io.StringIO(text)), **kw)
    else:
        atoms = PDBFile.read(io.StringIO(text)).get_structure(**kw)
    return to_pdb(atoms[struc.filter_amino_acids(atoms) | struc.filter_monoatomic_ions(atoms)])


# --- structure sources ----------------------------------------------------------------------
# Each kind has a fetcher returning DataFrame(structure_id, mol, protein_ref, [extra columns]) and a
# reader yielding the protein PDB text for a selection of those rows (only needed by write_structures).


def structures_zip(url, ligand_glob, protein_file="protein.pdb", id_pattern=r"([^/]+)/[^/]+$",
                   annotations=None, query=None, clean=False):
    """Ligand SDFs plus a protein file in the same folder, from a zip archive.

    protein_file: file next to each ligand; may contain ``{structure_id}``.
    id_pattern: regex whose first group, applied to the ligand path, is the structure ID.
    annotations: per-structure table in the archive to carry along, as
        ``{"file": <glob>, "id_column": <structure ID column>, "columns": [...]}``.
    query: pandas query on those columns to select structures (e.g. QC flags).
    clean: strip ligands/waters/additives from deposited protein files (see ``clean_protein``).
    """
    z = zipfile.ZipFile(download(url))
    rows = []
    for f in fnmatch.filter(z.namelist(), ligand_glob):
        sid = re.search(id_pattern, f).group(1)
        mol = Chem.MolFromMolBlock(z.read(f).decode(), removeHs=False)
        rows.append((sid, mol, f"{f.rsplit('/', 1)[0]}/{protein_file.format(structure_id=sid)}"))
    S = pd.DataFrame(rows, columns=["structure_id", "mol", "protein_ref"])
    if annotations:
        table = pd.read_csv(io.BytesIO(zip_member(z, annotations["file"])))
        table = table.rename(columns={annotations["id_column"]: "structure_id"})
        S = S.merge(table[["structure_id"] + annotations["columns"]], on="structure_id", how="left")
    return S.query(query).reset_index(drop=True) if query else S


def proteins_zip(S, url, clean=False, **_):
    z = zipfile.ZipFile(download(url))
    for ref in S.protein_ref:
        text = z.read(ref).decode()
        yield clean_protein(text) if clean else text


def _pdb_group_entries(group, additive_fraction=0.2):
    ids = json.loads(http_get(f"https://data.rcsb.org/rest/v1/core/entry_groups/{group}"))
    ids = ids["rcsb_group_container_identifiers"]["group_member_ids"]
    q = ("{entries(entry_ids:%s){rcsb_id struct{title} nonpolymer_entities{pdbx_entity_nonpoly{comp_id} "
         "nonpolymer_entity_instances{rcsb_nonpolymer_entity_instance_container_identifiers{asym_id}}}}}") % json.dumps(ids)
    entries = json.loads(http_get("https://data.rcsb.org/graphql", {"query": q}))["data"]["entries"]
    comps = {e["rcsb_id"]: [(n["pdbx_entity_nonpoly"]["comp_id"],
                             sorted(i["rcsb_nonpolymer_entity_instance_container_identifiers"]["asym_id"]
                                    for i in n["nonpolymer_entity_instances"]))
                            for n in e["nonpolymer_entities"] or []] for e in entries}
    # buffer/cryo additives (DMS, EDO, ...) show up in many entries of a group; ligands do not
    counts = pd.Series([c for v in comps.values() for c, _ in v]).value_counts()
    additives = set(counts[counts > additive_fraction * len(comps)].index)
    titles = {e["rcsb_id"]: e["struct"]["title"] for e in entries}
    return {pdb: [(c, a[0]) for c, a in v if c not in additives] for pdb, v in comps.items()}, titles


def structures_pdb_group(group, compound_id_pattern=None):
    """All ligand-bound entries of an RCSB PDB group deposition (e.g. ``G_1002283``).

    compound_id_pattern: regex that pulls the compound ID out of the entry title, used to pair
        structures with potencies by ID when the registered stereo of the potency sample is unreliable.
    """
    rows = []
    entries, titles = _pdb_group_entries(group)
    for pdb, ligands in sorted(entries.items()):
        m = compound_id_pattern and re.search(compound_id_pattern, titles[pdb])
        for comp, asym in ligands:
            sdf = http_get(f"https://models.rcsb.org/v1/{pdb}/ligand?label_comp_id={comp}"
                           f"&label_asym_id={asym}&encoding=sdf").decode()
            mol = Chem.MolFromMolBlock(sdf, removeHs=False)
            rows.append((f"{pdb}_{comp}_{asym}", m.group(1) if m else None, mol, pdb))
    return pd.DataFrame(rows, columns=["structure_id", "compound_id", "mol", "protein_ref"])


def proteins_pdb_group(S, **_):
    for pdb in S.protein_ref:
        yield clean_protein(http_get(f"https://files.rcsb.org/download/{pdb}.cif").decode(), fmt="cif")


def _polaris_group(dataset):
    import polaris  # noqa: F401  (registers the Polaris zarr codecs; `pip install polaris-lib`)
    import zarr

    return zarr.open_consolidated(f"https://data.polarishub.io/dataset/{dataset}/data.zarr", mode="r")


def _polaris_poses(dataset, filter):
    g = _polaris_group(dataset)
    keep = np.ones(g[list(filter)[0]].shape[0], bool)
    for col, value in filter.items():
        keep &= g[col][:] == value
    idx = np.flatnonzero(keep)
    return pd.DataFrame({"structure_id": [f"{dataset.split('/')[-1]}-{i}" for i in idx],
                         "mol": list(g["Ligand Pose"].oindex[idx]), "protein_ref": idx})


def structures_polaris(dataset, filter):
    """Ligand poses (and protein structures) from a Polaris dataset, e.g. filter={"Set": "Test"}."""
    return cached(_polaris_poses, dataset=dataset, filter=filter)


def proteins_polaris(S, dataset, **_):
    g = _polaris_group(dataset)
    idx = np.sort(S.protein_ref.to_numpy())
    atoms = dict(zip(idx, g["Protein Structure"].oindex[idx]))
    for i in S.protein_ref:
        yield to_pdb(atoms[i])


STRUCTURES = {
    "zip": (structures_zip, proteins_zip),
    "pdb_group": (structures_pdb_group, proteins_pdb_group),
    "polaris": (structures_polaris, proteins_polaris),
}


# --- potency sources: each returns DataFrame(compound_id, smiles, potency_type, relation, pX) ----


def to_px(values, unit):
    """Convert concentrations in ``unit`` (or 'pX' values) to -log10(M)."""
    values = pd.to_numeric(values, errors="coerce")
    return values if unit == "pX" else -np.log10(values * UNIT_TO_M[unit])


def potency_csv(url, file, columns, unit, potency_type="IC50"):
    """A CSV table (in a GitHub repo, or inside a zip archive at ``url``).

    columns: maps source columns to compound_id / smiles / value.
    unit: unit of ``value`` (M, mM, uM, nM) or 'pX' for -log10 values. Values may carry a
        qualifier ("> 99.5"), which ends up in ``relation``.
    """
    df = read_csv(url, file).rename(columns=columns)
    value = df["value"].astype(str).str.strip()
    df["relation"] = value.str.extract(r"^([<>]=?)")[0].fillna("=")
    df["pX"] = to_px(value.str.lstrip("<>= "), unit)
    return df.assign(potency_type=potency_type)[["compound_id", "smiles", "potency_type", "relation", "pX"]]


def _chembl_pages(endpoint, key, **params):
    rows, url = [], f"/chembl/api/data/{endpoint}.json?" + "&".join(f"{k}={v}" for k, v in params.items())
    while url:
        page = json.loads(http_get("https://www.ebi.ac.uk" + url))
        rows += page[key]
        url = page["page_meta"]["next"]
    return rows


def potency_chembl(document, potency_type="IC50"):
    """All activities of one type from a ChEMBL document (e.g. an ASAP deposition ``CHEMBL5441456``).

    ``compound_id`` is the depositor's compound key (e.g. an ASAP ID) where available.
    """
    df = pd.DataFrame(_chembl_pages("activity", "activities", document_chembl_id=document,
                                    standard_type=potency_type, limit=1000))
    df = df[df.standard_units == "nM"]
    keys = {r["record_id"]: r["compound_key"]
            for r in _chembl_pages("compound_record", "compound_records", document_chembl_id=document, limit=1000)}
    return pd.DataFrame({
        "compound_id": df.record_id.map(keys).fillna(df.molecule_chembl_id),
        "smiles": df.canonical_smiles, "potency_type": potency_type,
        "relation": df.standard_relation, "pX": to_px(df.standard_value, "nM"),
    })


def _polaris_potency(dataset, column, id_column, potency_type):
    g = _polaris_group(dataset)
    return pd.DataFrame({"compound_id": g[id_column][:], "smiles": g["CXSMILES"][:],
                         "potency_type": potency_type, "relation": "=", "pX": g[column][:]}).dropna(subset=["pX"])


def potency_polaris(dataset, column, id_column="Molecule Name", potency_type="IC50"):
    """One pX column (e.g. 'pIC50 (SARS-CoV-2 Mpro)') of a Polaris dataset."""
    return cached(_polaris_potency, dataset=dataset, column=column, id_column=id_column, potency_type=potency_type)


POTENCY = {"csv": potency_csv, "chembl": potency_chembl, "polaris": potency_polaris}


# --- building and loading -----------------------------------------------------------------


def dataset_dir(name):
    return DATA_DIR / re.sub(r"\W+", "_", name).strip("_").lower()


def _structures(src):
    spec = dict(src["structures"])
    fetch, read_proteins = STRUCTURES[spec.pop("from")]
    return fetch(**spec), read_proteins, spec


def _potency(src):
    spec = dict(src["potency"])
    return POTENCY[spec.pop("from")](**spec)


def build_pairs(name, src):
    """Fetch structures and potencies, keep compounds with both and write ``pairs.csv``."""
    S, _, _ = _structures(src)
    P = _potency(src)
    S["key"] = S.mol.map(match_key)
    mols = P.smiles.map(lambda s: Chem.MolFromSmiles(s) if isinstance(s, str) else None)
    P = P.assign(key=mols.map(match_key), potency_stereo=mols.map(lambda m: m and stereo_label(m)))
    # censored values (">100 uM") are not usable measurements; average repeats per compound
    P = P[(P.relation == "=") & P.pX.notna() & P.key.notna()].drop_duplicates(["compound_id", "key", "pX"])
    agg = dict(compound_id=("compound_id", "first"), smiles=("smiles", "first"),
               potency_type=("potency_type", "first"), potency_stereo=("potency_stereo", "first"),
               pX=("pX", "mean"), n_measurements=("pX", "size"))
    exact = S.merge(P.groupby("key").agg(**agg).reset_index(), on="key")
    # a crystal shows one stereoisomer; if the measured sample left its stereo unassigned, match on connectivity
    S["skeleton"], P["skeleton"] = S.key.str[:14], P.key.str[:14]
    loose = P[P.potency_stereo == "unspecified"].groupby("skeleton").agg(**agg).reset_index()
    loose = S[~S.structure_id.isin(exact.structure_id)].merge(loose, on="skeleton")
    pairs = [exact, loose]
    # structures that carry a compound ID: pair by ID (same connectivity), trusting the crystal's stereo
    if "compound_id" in S:
        by_id = P.groupby("compound_id").agg(**{k: v for k, v in agg.items() if k != "compound_id"},
                                             skeleton=("skeleton", "first")).reset_index()
        rest = S[~S.structure_id.isin(exact.structure_id) & ~S.structure_id.isin(loose.structure_id)]
        pairs.append(rest.merge(by_id, on=["compound_id", "skeleton"]).assign(potency_stereo="id_match"))
        pairs = [p.drop(columns="compound_id_x", errors="ignore").rename(columns={"compound_id_y": "compound_id"})
                 for p in pairs]
    pairs = pd.concat(pairs).sort_values("structure_id").reset_index(drop=True)
    pairs["expt_dg"] = -RT * np.log(10) * pairs.pX  # exact for Kd; assumes IC50 ~ Kd otherwise
    pairs = pairs.assign(dataset=name, **src.get("meta", {}))
    extra = [c for c in pairs.columns if c not in PAIR_COLUMNS + INTERNAL_COLUMNS]

    out = dataset_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    pairs[PAIR_COLUMNS + extra].to_csv(out / "pairs.csv", index=False)
    (out / "source.json").write_text(json.dumps(src, indent=2, sort_keys=True))
    print(f"{name}: {len(S)} structures, {P.key.nunique()} compounds with potency -> "
          f"{len(pairs)} structures of {pairs.compound_id.nunique()} compounds with both")


def load_all(sources, rebuild=False):
    """Load every SOURCES entry into one table of structure-potency pairs, (re)building a
    dataset when it is missing or its SOURCES entry changed."""
    frames = []
    for name, src in sources.items():
        out = dataset_dir(name)
        current = json.dumps(src, indent=2, sort_keys=True)
        if rebuild or not (out / "source.json").exists() or (out / "source.json").read_text() != current:
            build_pairs(name, src)
        frames.append(pd.read_csv(out / "pairs.csv").assign(dataset=name))
    return pd.concat(frames, ignore_index=True)


def write_structures(sources, overwrite=False):
    """Write the crystal ligand and protein of every structure-potency pair to
    ``data/<name>/structures/<structure_id>/{ligand.sdf,protein.pdb}`` (skipping existing ones)."""
    for name, src in sources.items():
        pairs = load_all({name: src}).set_index("structure_id")
        S, read_proteins, spec = _structures(src)
        S = S[S.structure_id.isin(pairs.index)]
        out = dataset_dir(name) / "structures"
        todo = S[[overwrite or not (out / sid / "protein.pdb").exists() for sid in S.structure_id]]
        for row, pdb in zip(todo.itertuples(), read_proteins(todo, **spec)):
            folder = out / row.structure_id
            folder.mkdir(parents=True, exist_ok=True)
            mol = Chem.Mol(row.mol)
            mol.SetProp("_Name", row.structure_id)
            for col in ["compound_id", "smiles", "potency_type", "pX", "expt_dg"]:
                mol.SetProp(col, str(pairs.at[row.structure_id, col]))
            with Chem.SDWriter(str(folder / "ligand.sdf")) as w:
                w.write(mol)
            (folder / "protein.pdb").write_text(pdb)
        print(f"{name}: {len(S)} structures in {out} ({len(todo)} written)")


# --- analysis -------------------------------------------------------------------------------

# Chemical motifs where off-atom charges (virtual sites) are commonly argued to matter.
VSITE_MOTIFS = {
    "aromatic Cl/Br/I (sigma hole)": "c[Cl,Br,I]",
    "aliphatic Cl/Br/I": "[CX4][Cl,Br,I]",
    "aromatic N acceptor (pyridine-like)": "[nX2]",
    "carbonyl O": "[CX3]=[OX1]",
    "ether / hydroxyl O": "[OX2;!$(O-C=O)]",
    "nitrile N": "C#N",
    "divalent S": "[SX2]",
}

def summarise(df):
    """Per-dataset counts of structures, unique compounds and the experimental affinity range."""
    return df.groupby("dataset", sort=False).agg(
        potency_type=("potency_type", "first"),
        n_structures=("structure_id", "size"),
        n_compounds=("compound_id", "nunique"),
        pX_min=("pX", "min"),
        pX_max=("pX", "max"),
        expt_dg_min=("expt_dg", "min"),
        expt_dg_max=("expt_dg", "max"),
    ).round(2)


def motif_counts(df, motifs=VSITE_MOTIFS):
    """Number and fraction of unique compounds per dataset containing each vsite-relevant motif."""
    unique = df.drop_duplicates(["dataset", "compound_id"]).copy()
    mols = unique.smiles.map(Chem.MolFromSmiles)
    patterns = {k: Chem.MolFromSmarts(v) for k, v in motifs.items()}
    for k, patt in patterns.items():
        unique[k] = [m is not None and m.HasSubstructMatch(patt) for m in mols]
    counts = unique.groupby("dataset", sort=False)[list(patterns)].sum().T
    fracs = unique.groupby("dataset", sort=False)[list(patterns)].mean().T
    return pd.concat({"n": counts, "frac": fracs.round(2)}, axis=1)
