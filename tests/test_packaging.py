"""Check delivery boundaries and the test gate before publishing an archive."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import package_project as packager


class PackagingChecks(unittest.TestCase):
    def test_delivery_selection_excludes_local_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cfg={'name':'Test','student_id':'123'}
            (root/'project_config.json').write_text(json.dumps(cfg),encoding='utf-8')
            deliverables={
                'project_config.json','README.md','src/data.py','tests/test_project.py',
                'data/raw/beijing-air-quality.zip','data/processed/daily.csv',
                'results/verification.json','results/figures/01_missing.png',
                'report/Test123期末大作业.pdf',
            }
            local_only={
                'assignment.pdf','notes.txt','tmp/audit.py','_review_tmp/page.png',
                '.git/config','.venv/pyvenv.cfg','src/__pycache__/data.pyc',
                'report/修订版/Test123期末大作业.pdf','report/draft.pdf',
                'output/python_final_2026/old.zip','results/old.zip',
            }
            for name in (deliverables|local_only)-{'project_config.json'}:
                path=root/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b'fixture')
            with mock.patch.object(packager,'ROOT',root):
                selected={p.relative_to(root).as_posix() for p in packager.files()}
            self.assertEqual(selected,deliverables)

    def test_test_summary_is_measured_and_failures_are_rejected(self):
        def fail():
            raise AssertionError('intentional failed check')

        def error():
            raise ValueError('intentional error')

        def skip():
            raise unittest.SkipTest('intentional skip')

        cases=[
            ('success',[lambda: None,skip]),
            ('failure',[fail]),('error',[error]),('empty',[]),
        ]
        for name,checks in cases:
            with self.subTest(name=name):
                suite=unittest.TestSuite(unittest.FunctionTestCase(f) for f in checks)
                runner=unittest.TextTestRunner(stream=io.StringIO())
                with mock.patch.object(unittest.TestLoader,'discover',return_value=suite), \
                     mock.patch.object(unittest,'TextTestRunner',return_value=runner):
                    if name=='success':
                        summary=packager.run_unit_tests()
                        self.assertEqual(summary['status'],'passed')
                        self.assertEqual(summary['tests_run'],2)
                        self.assertEqual(summary['skipped'],1)
                        self.assertEqual(summary['failures'],0)
                        self.assertEqual(summary['errors'],0)
                    else:
                        with self.assertRaisesRegex(RuntimeError,'Packaging stopped'):
                            packager.run_unit_tests()

    def test_failed_tests_stop_before_writing_results_or_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            verification=root/'results/verification.json'
            verification.parent.mkdir()
            verification.write_text('previous verification',encoding='utf-8')
            with mock.patch.object(packager,'ROOT',root), \
                 mock.patch.object(packager,'run_unit_tests',side_effect=RuntimeError('failed')):
                with self.assertRaisesRegex(RuntimeError,'failed'):
                    packager.package(root/'delivery')
            self.assertFalse((root/'delivery').exists())
            self.assertEqual(verification.read_text(encoding='utf-8'),'previous verification')


if __name__=='__main__':
    unittest.main()
