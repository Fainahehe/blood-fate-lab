import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from train import examples, fit_predict, metrics, load, PREFIX
import tempfile, gzip
from pathlib import Path
from scipy.io import mmwrite

class BaselineTests(unittest.TestCase):
    def test_original_format_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            m=pd.DataFrame({'Time point':[2,6], 'Cell type annotation':['Undifferentiated','Monocyte']})
            m.to_csv(d/(PREFIX+'metadata.txt.gz'),sep='\t',index=False)
            with gzip.open(d/(PREFIX+'gene_names.txt.gz'),'wt') as f: f.write('GeneA\nGeneB\n')
            for suffix, arr in [('normed_counts.mtx.gz',[[2.,3.],[9.,8.]]),('clone_matrix.mtx.gz',[[1],[1]])]:
                with gzip.open(d/(PREFIX+suffix),'wb') as f: mmwrite(f,sparse.coo_matrix(arr))
            X,C,meta,genes=load(d)
            np.testing.assert_allclose(X.toarray(),[[2,3],[0,0]])
            self.assertEqual(list(genes),['GeneA','GeneB'])
            self.assertEqual(C.shape,(2,1))

    def test_mixed_targets_and_early_only_inputs(self):
        X = sparse.csr_matrix([[1., 0], [999, 888], [777, 666], [2, 0], [0, 4]])
        C = sparse.csr_matrix([[1,0], [1,0], [1,0], [0,1], [0,1]])
        m = pd.DataFrame({'Time point':[2,6,6,2,6],
                          'Cell type annotation':['Undifferentiated','Neutrophil','Monocyte','Undifferentiated','Monocyte']})
        E, t, _ = examples(X,C,m,1)
        np.testing.assert_allclose(t.target_neutrophil_fraction,[.5,0])
        np.testing.assert_allclose(E.toarray(),np.log1p([[1,0],[2,0]]))
        C[0,1]=1
        E,t,a = examples(X,C,m,1)
        self.assertEqual(len(t),0)
        self.assertEqual(a['ambiguous_clone_columns_excluded'],2)

    def test_fractional_model_and_test_only_feature(self):
        rng=np.random.default_rng(5)
        X=rng.random((100,8)); X[:80,7]=0; X[80:,7]=10000
        y=1/(1+np.exp(-8*(X[:,0]-.5)))
        pred, fitted=fit_predict(sparse.csr_matrix(X[:80]),y[:80],sparse.csr_matrix(X[80:]),5)
        self.assertTrue(np.all((pred>=0)&(pred<=1)))
        self.assertFalse(fitted[0].get_support()[7])
        self.assertLess(metrics(y[80:],pred)['clone_macro_mae'], metrics(y[80:],np.full(20,y[:80].mean()))['clone_macro_mae'])

if __name__=='__main__': unittest.main()
