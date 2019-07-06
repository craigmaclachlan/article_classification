import os
import stat
import shutil
import unittest
import tempfile
import numpy
import logging

logger = logging.getLogger()

from articlass.model import ModelConfiguration, ClassifierModel, pred2str

class TestModelCfg(unittest.TestCase):
    """Test the model config class."""

    def setUp(self):
        self.dirpath = tempfile.mkdtemp()

        self.m_cfg = ModelConfiguration()
        self.m_cfg.setinfo(self.dirpath,
                           ['a', 'b', 'c'],
                           5.0,
                           ['cat', 'dog', 'bear'],
                           model_name='animals'
                           )

    def tearDown(self):
        shutil.rmtree(self.dirpath)

    def assertClassEqual(self, cleft, cright):
        self.assertEqual(cleft.classes, cright.classes)
        self.assertEqual(cleft.dirpath, cright.dirpath)
        self.assertEqual(cleft.conf_path, cright.conf_path)
        self.assertEqual(cleft.model_path, cright.model_path)
        self.assertEqual(cleft.fig_path, cright.fig_path)
        # Because the log file is dated it is a bit difficult to
        # reliably test.
        self.assertEqual(cleft.log_path, cright.log_path)
        self.assertEqual(cleft.norm, cright.norm)
        self.assertEqual(cleft.terms, cright.terms)


    def test_setinfo(self):

        self.assertEqual(['a', 'b', 'c'], self.m_cfg.classes)
        self.assertEqual(self.dirpath, self.m_cfg.dirpath)
        self.assertEqual(os.path.join(self.dirpath, 'animals.json'),
                         self.m_cfg.conf_path)
        self.assertEqual(os.path.join(self.dirpath, 'animals.h5'),
                         self.m_cfg.model_path)
        self.assertEqual(os.path.join(self.dirpath, 'animals.png'),
                         self.m_cfg.fig_path)
        # Because the log file is dated it is a bit difficult to
        # reliably test.
        self.assertTrue(
            self.m_cfg.log_path.startswith(
                os.path.join(self.dirpath, 'log-')))
        self.assertEqual(5.0, self.m_cfg.norm)
        self.assertEqual(['cat', 'dog', 'bear'], self.m_cfg.terms)

    def test_invalid_norm(self):
        m_cfg = ModelConfiguration()
        with self.assertRaises(TypeError):
            m_cfg.setinfo('/mymodel',
                          ['a', 'b', 'c'],
                          'notanumber',
                          ['cat', 'dog', 'bear'],
                          model_name='animals'
                          )

    def test_invalid_classlist(self):
        m_cfg = ModelConfiguration()
        with self.assertRaises(TypeError):
            m_cfg.setinfo('/mymodel',
                          'my classes are a b c',
                          5,
                          ['cat', 'dog', 'bear'],
                          model_name='animals'
                          )

    def test_invalid_wordlist(self):
        m_cfg = ModelConfiguration()
        with self.assertRaises(TypeError):
            m_cfg.setinfo('/mymodel',
                          ['a', 'b', 'c'],
                          5,
                          {1: 'cat', 2: 'dog', 3: 'bear'},
                          model_name='animals'
                          )

    def test_save_and_load(self):
        good_savepath = self.m_cfg.save()
        self.assertEqual(good_savepath, self.m_cfg.conf_path)

        m_cfg = ModelConfiguration()
        m_cfg.load(good_savepath)
        self.assertClassEqual(m_cfg, self.m_cfg)

    def test_save_no_permision(self):
        os.chmod(self.dirpath, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        with self.assertRaises(IOError):
            self.m_cfg.save()

    def test_save_mkdir(self):
        shutil.rmtree(self.dirpath)
        good_savepath = self.m_cfg.save()
        self.assertEqual(good_savepath, self.m_cfg.conf_path)


class TestClassifierModel(unittest.TestCase):
    """Test the class wrapping the model training and execution."""

    def test_not_yet_defined(self):
        model = ClassifierModel()
        with self.assertRaises(RuntimeError):
            model.train(1,1, 'mypath')

    def test_not_yet_trained_test(self):
        model = ClassifierModel()
        with self.assertRaises(RuntimeError):
            model.test(1,1)

    def test_not_yet_trained_pred(self):
        model = ClassifierModel()
        with self.assertRaises(RuntimeError):
            model.predict(1)

    def test_not_yet_trained_stats(self):
        model = ClassifierModel()
        with self.assertRaises(RuntimeError):
            model.stats()

    def test_not_yet_trained_plot(self):
        model = ClassifierModel()
        with self.assertRaises(RuntimeError):
            model.export_plot('mypath')

    def test_not_yet_trained_save(self):
        model = ClassifierModel()
        with self.assertRaises(RuntimeError):
            model.save('mypath')


class TestPredPrint(unittest.TestCase):

    def setUp(self):
        self.predictions = numpy.array([0.1, 0.18995, 0.01005, 0.69])
        self.classes = ['cat', 'dog', 'bear', 'tiger']

    def test_normal(self):
        expected = "bear : 0.010\ncat : 0.100\ndog : 0.190\ntiger : 0.690"
        result = pred2str(self.predictions, self.classes)
        self.assertEqual(expected, result)

    def test_full_precedence(self):
        expected = "bear : 0.010\ncat : 0.100\ndog : 0.190\ntiger : 0.690"
        result = pred2str(self.predictions, self.classes, full=True,
                          class_only=True)
        self.assertEqual(expected, result)

    def test_notfull(self):
        expected = "tiger : 0.690"
        result = pred2str(self.predictions, self.classes, full=False)
        self.assertEqual(expected, result)

    def test_classonly(self):
        expected = "tiger"
        result = pred2str(self.predictions, self.classes, class_only=True,
                          full=False)
        self.assertEqual(expected, result)

    def test_invalid_preds(self):
        with self.assertRaises(TypeError):
            pred2str('abc', self.classes)
        with self.assertRaises(TypeError):
            pred2str(['a', 0.9, 0.8], self.classes)

    def test_invalid_classes(self):
        with self.assertRaises(TypeError):
            pred2str(self.predictions, 'abc')

    def test_unequal_lengths(self):
        with self.assertRaises(IndexError):
            pred2str(self.predictions, self.classes[:-2])