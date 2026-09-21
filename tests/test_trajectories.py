import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('prepare', Path(__file__).resolve().parents[1]/'scripts/prepare_trajectories.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_future_context_is_annotation_only_and_attempts_stay_together():
    turns = [{'role':'system', 'text':'setup'}, {'role':'user', 'text':'Fix parser failure'},
             {'role':'ai','text':'Inspect failures\n```bash\npytest tests/test_parse.py\n```'},
             {'role':'user','text':'AssertionError at parse.py:42'},
             {'role':'ai','text':'Future answer'}]
    source = {'instance_id':'owner__repo-12', 'trajectory':turns,'target':True,'model_name':'model'}
    record = list(prepare.records(source))[0]
    assert record['objective'] == 'Fix parser failure'
    assert record['command'] == 'pytest tests/test_parse.py'
    assert record['state'] == 'AssertionError at parse.py:42'
    assert record['group'] == 'owner/repo'
    assert record['label_status'] == 'unlabeled'
    assert record['annotation_only']['next_agent_message'] == 'Future answer'
    source['instance_id'] = 'owner__repo-14'
    other = list(prepare.records(source))[0]
    assert prepare.split_for(record['group']) == prepare.split_for(other['group'])
