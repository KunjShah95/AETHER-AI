
import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terminal.code_agent import CodeAgent
from terminal.workflow_engine import WorkflowEngine, StepType
from terminal.pair_programmer import PairProgrammer
# SmartRAG requires sentence-transformers which might not be installed in test env
try:
    from terminal.smart_rag import SmartRAG
except ImportError:
    SmartRAG = None

class TestCodeAgent(unittest.TestCase):
    def setUp(self):
        self.mock_ai = MagicMock(return_value="AI Response")
        self.agent = CodeAgent(ai_query_func=self.mock_ai)
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_safe_path(self):
        # Should resolve to absolute path in cwd
        path = self.agent._resolve_path("test.py")
        self.assertTrue(os.path.isabs(path))
        
        # Should block traversal
        with self.assertRaises(ValueError):
            self.agent._resolve_path("../../../etc/passwd")
            
    def test_is_safe_file(self):
        self.assertTrue(self.agent._is_safe_file("test.py"))
        self.assertTrue(self.agent._is_safe_file("test.js"))
        self.assertTrue(self.agent._is_safe_file("README.md"))
        self.assertFalse(self.agent._is_safe_file("test.exe"))
        self.assertFalse(self.agent._is_safe_file(".env"))
        
    def test_analyze_project(self):
        # Create some dummy files
        os.makedirs(os.path.join(self.test_dir, "src"))
        with open(os.path.join(self.test_dir, "main.py"), "w") as f:
            f.write("print('hello')\n# TODO: fix this")
        
        analysis = self.agent.analyze_project(self.test_dir)
        
        self.assertEqual(analysis['files'], 1)
        self.assertEqual(analysis['languages']['Python'], 1)
        self.assertEqual(analysis['issues_count'], 1)
        
    def test_create_edit_from_ai(self):
        # Mock AI response to return a valid diff
        self.mock_ai.return_value = """Here is the fix:
```python
<<<<<<< SEARCH
def hello():
    pass
=======
def hello():
    print("world")
>>>>>>> REPLACE
```"""
        
        # Create dummy file
        filepath = os.path.join(self.test_dir, "test.py")
        with open(filepath, "w") as f:
            f.write("def hello():\n    pass")
            
        edit = self.agent.create_edit_from_ai(filepath, "Make it print world")
        
        self.assertIsNotNone(edit)
        self.assertIn('print("world")', edit.new_content)

class TestWorkflowEngine(unittest.TestCase):
    def setUp(self):
        self.mock_ai = MagicMock(return_value="AI Response")
        self.engine = WorkflowEngine(base_dir=tempfile.mkdtemp(), ai_query_func=self.mock_ai)
        
    def tearDown(self):
        shutil.rmtree(self.engine.base_dir)
        
    def test_create_workflow(self):
        wf = self.engine.create_workflow("Test Workflow")
        self.assertEqual(wf.name, "Test Workflow")
        self.assertEqual(wf.status.value, "pending")
        
    def test_add_step(self):
        wf = self.engine.create_workflow("Test Workflow")
        self.engine.add_step(wf.workflow_id, StepType.COMMAND, "Echo", {"command": "echo hello"})
        
        self.assertEqual(len(wf.steps), 1)
        self.assertEqual(wf.steps[0].name, "Echo")
        
    def test_run_workflow(self):
        wf = self.engine.create_workflow("Test Run")
        self.engine.add_step(wf.workflow_id, StepType.COMMAND, "Echo", {"command": "echo hello"})
        
        # Mock subprocess run if needed, but echo is safe
        result = self.engine.run_workflow(wf.workflow_id)
        
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(result['results']), 1)
        
    def test_variable_substitution(self):
        wf = self.engine.create_workflow("Var Test")
        self.engine.add_step(wf.workflow_id, StepType.COMMAND, "Echo Var", {"command": "echo ${greeting}"})
        
        # We can't easily check stdout of subprocess here without mocking, 
        # but we can test the substitution logic directly
        config = {"msg": "Hello ${name}"}
        vars = {"name": "World"}
        substituted = self.engine._substitute_variables(config, vars)
        self.assertEqual(substituted['msg'], "Hello World")

class TestPairProgrammer(unittest.TestCase):
    def setUp(self):
        self.mock_ai = MagicMock()
        self.pair = PairProgrammer(ai_query_func=self.mock_ai)
        
    def test_start_session(self):
        session = self.pair.start_session("test.py")
        self.assertEqual(session.filepath, "test.py")
        self.assertEqual(session.language, "python")
        self.assertIn("python", self.pair.active_sessions)
        
    def test_suggest_completion(self):
        self.mock_ai.return_value = "def world():\n    return 'world'"
        
        suggestions = self.pair.suggest_completion("def hello():\n    pass\n", 20)
        
        self.assertTrue(len(suggestions) > 0)
        self.assertEqual(suggestions[0].type, "completion")

if __name__ == '__main__':
    unittest.main()
