"""Clone-level LARRY baseline. See README for scientific assumptions."""
import argparse
import gzip
import hashlib
import json
import platform
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.io import mmread
from sklearn import __version__ as sklearn_version
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

BASE = 'https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/'
PREFIX = 'stateFate_inVitro_'
SUFFIXES = ['normed_counts.mtx.gz', 'gene_names.txt.gz', 'clone_matrix.mtx.gz', 'metadata.txt.gz']


def download(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in SUFFIXES:
        path = folder / (PREFIX + suffix)
        if path.exists():
            continue
        print('Downloading', path.name, flush=True)
        temporary = path.with_suffix('.partial')
        try:
            with urllib.request.urlopen(BASE + path.name, timeout=120) as response, temporary.open('wb') as out:
                import shutil
                shutil.copyfileobj(response, out)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def load(folder):
    def matrix(suffix):
        with gzip.open(folder / (PREFIX + suffix), 'rb') as f:
            return sparse.csr_matrix(mmread(f), dtype=np.float64)
    meta = pd.read_csv(folder / (PREFIX + 'metadata.txt.gz'), sep='\t')
    genes = pd.read_csv(folder / (PREFIX + 'gene_names.txt.gz'), header=None)[0].astype(str).to_numpy()
    clones = matrix('clone_matrix.mtx.gz')
    # The full original count file is ~2 GB compressed. Stream it and retain
    # early-cell entries only; later RNA must never be used as model input.
    early_rows = pd.to_numeric(meta['Time point'], errors='raise').to_numpy() == 2
    rows, cols, vals = [], [], []
    with gzip.open(folder / (PREFIX + 'normed_counts.mtx.gz'), 'rt') as f:
        line = f.readline()
        if not line.startswith('%%MatrixMarket matrix coordinate'):
            raise ValueError('Expected coordinate Matrix Market format.')
        line = f.readline()
        while line.startswith('%'):
            line = f.readline()
        nr, nc, nnz = map(int, line.split())
        if (nr, nc) != (len(meta), len(genes)):
            raise ValueError('Expression dimensions do not match metadata/genes.')
        seen = 0
        for chunk in pd.read_csv(f, sep=r'\s+', header=None, names=['row','col','value'],
                                 dtype={'row':np.int32,'col':np.int32,'value':np.float32},
                                 chunksize=1000000):
            r = chunk.row.to_numpy()-1
            c = chunk.col.to_numpy()-1
            v = chunk.value.to_numpy()
            if (r < 0).any() or (r >= nr).any() or (c < 0).any() or (c >= nc).any():
                raise ValueError('Out-of-range matrix index.')
            if not np.isfinite(v).all() or (v < 0).any():
                raise ValueError('Invalid RNA counts.')
            keep = early_rows[r]
            rows.append(r[keep]); cols.append(c[keep]); vals.append(v[keep])
            seen += len(chunk)
        if seen != nnz:
            raise ValueError('Truncated count matrix.')
    X = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nr,nc))
    if X.shape != (len(meta), len(genes)) or clones.shape[0] != len(meta):
        raise ValueError('Expression, genes, metadata and clone matrix are misaligned.')
    for col in ['Time point', 'Cell type annotation']:
        if col not in meta:
            raise ValueError(f'Missing {col!r}; available columns: {list(meta.columns)}')
    if not np.isfinite(X.data).all() or (X.data < 0).any():
        raise ValueError('Expected finite non-negative normalized counts.')
    return X, clones, meta, genes


def examples(X, clones, meta, min_late=5):
    """Discard ambiguous clone columns entirely; aggregate day-2 expression by clone."""
    C = clones.copy().tocsr()
    C.eliminate_zeros()
    C.data[:] = 1
    ambiguous = np.asarray(C.sum(axis=1)).ravel() > 1
    bad = np.asarray(C[ambiguous].sum(axis=0)).ravel() > 0
    C = C[:, ~bad]
    clone_ids = np.flatnonzero(~bad)
    day = pd.to_numeric(meta['Time point'], errors='raise').to_numpy()
    labels = meta['Cell type annotation'].astype(str).str.lower().to_numpy()
    early = day == 2
    neu = (day == 6) & (labels == 'neutrophil')
    mon = (day == 6) & (labels == 'monocyte')
    n_early = np.asarray(C[early].sum(axis=0)).ravel()
    n_neu = np.asarray(C[neu].sum(axis=0)).ravel()
    n_mon = np.asarray(C[mon].sum(axis=0)).ravel()
    n_all = np.asarray(C[day == 6].sum(axis=0)).ravel()
    keep = (n_early > 0) & (n_neu + n_mon >= min_late)
    transformed = X[early].copy()
    transformed.data = np.log1p(transformed.data)
    E = sparse.diags(1 / n_early[keep]) @ C[early][:, keep].T @ transformed
    table = pd.DataFrame({'clone_id': clone_ids[keep], 'early_cells': n_early[keep],
                          'neutrophils': n_neu[keep], 'monocytes': n_mon[keep],
                          'all_day6_cells': n_all[keep]})
    table['target_neutrophil_fraction'] = table.neutrophils / (table.neutrophils + table.monocytes)
    table['target_lineage_coverage'] = (table.neutrophils + table.monocytes) / table.all_day6_cells
    audit = {'cells': X.shape[0], 'genes': X.shape[1], 'original_clones': clones.shape[1],
             'ambiguous_clone_columns_excluded': int(bad.sum()), 'eligible_clones': int(keep.sum()),
             'time_counts': meta['Time point'].value_counts().to_dict(),
             'state_counts': meta['Cell type annotation'].value_counts().to_dict()}
    return E.tocsr(), table, audit


def fit_predict(Xtrain, ytrain, Xtest, n_genes=500):
    # Every learned transform sees training clones ONLY.
    varying = VarianceThreshold(0)
    A = varying.fit_transform(Xtrain)
    B = varying.transform(Xtest)
    select = SelectKBest(f_regression, k=min(n_genes, A.shape[1]))
    A = select.fit_transform(A, ytrain)
    B = select.transform(B)
    # Fractional targets: duplicate each clone with class weights y and 1-y.
    # Total training weight is one per clone, regardless of its sampled size.
    model = LogisticRegression(C=0.1, solver='lbfgs', max_iter=3000)
    model.fit(sparse.vstack([A, A]), np.r_[np.ones(len(ytrain)), np.zeros(len(ytrain))],
              sample_weight=np.r_[ytrain, 1-ytrain])
    return model.predict_proba(B)[:, 1], (varying, select, model)


def metrics(y, pred):
    p = np.clip(pred, 1e-8, 1-1e-8)
    return {'clone_macro_mae': float(np.mean(abs(y-p))),
            'clone_macro_rmse': float(np.sqrt(np.mean((y-p)**2))),
            'fractional_cross_entropy': float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data'))
    p.add_argument('--output-dir', type=Path, default=Path('results'))
    p.add_argument('--download', action='store_true')
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--min-late', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.min_late < 1:
        p.error('--min-late must be positive')
    if args.download:
        download(args.data_dir)
    X, C, meta, genes = load(args.data_dir)
    E, table, audit = examples(X, C, meta, args.min_late)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    audit['versions'] = {'python': platform.python_version(), 'numpy': np.__version__,
                         'scipy': scipy.__version__, 'sklearn': sklearn_version}
    audit['file_sha256'] = {}
    for suffix in SUFFIXES:
        path = args.data_dir / (PREFIX + suffix)
        with path.open('rb') as f:
            audit['file_sha256'][path.name] = hashlib.file_digest(f, 'sha256').hexdigest()
    (out / 'audit.json').write_text(json.dumps(audit, indent=2, default=str))
    table.to_csv(out / 'clone_outcomes.csv', index=False)
    print(json.dumps(audit, indent=2, default=str))
    if args.audit_only:
        return
    if len(table) < 30:
        raise ValueError('Fewer than 30 eligible clones; inspect audit before modeling.')
    train, test = train_test_split(np.arange(len(table)), test_size=0.25, random_state=args.seed)
    y = table.target_neutrophil_fraction.to_numpy()
    if np.std(y[train]) < 1e-10:
        raise ValueError('No target variation in training clones.')
    pred, fitted = fit_predict(E[train], y[train], E[test])
    baseline = np.full(len(test), y[train].mean())
    report = {'seed': args.seed, 'min_late': args.min_late, 'train_clones': len(train),
              'test_clones': len(test), 'model': metrics(y[test], pred), 'baseline': metrics(y[test], baseline),
              'scope': 'Held-out clones in the same pooled experiment setting; NOT donor/lab generalization.'}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2))
    table['split'] = 'train'
    table.loc[test, 'split'] = 'test'
    table.to_csv(out / 'clone_outcomes.csv', index=False)
    prediction = table.iloc[test].copy()
    prediction['predicted_neutrophil_fraction'] = pred
    prediction['baseline_neutrophil_fraction'] = baseline
    prediction.to_csv(out / 'test_predictions.csv', index=False)
    import joblib
    joblib.dump({'transforms_model': fitted, 'genes': genes, 'seed': args.seed,
                 'preprocessing': 'mean of log1p normalized counts across early cells in clone'}, out / 'model.joblib')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
