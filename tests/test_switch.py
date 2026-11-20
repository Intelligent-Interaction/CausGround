"""CPU-only SWITCH public contracts; run with python -I -B tests/test_switch.py."""
import contextlib
import copy
import io
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from causground import CausGround
from causground.query import main as shared_main
from causground.switch import SwitchCWM
from causground.switch.__main__ import main as standalone_main

VERSIONS = ('simple', 'complex')
NO_MATCH = 'zzzzqqqxxxvvv'
STRICT_HEADER = (
    'Observed preconditions and effects are summarized from interaction trajectories.\n'
    'When only a few trajectories are available for an action, these summaries may be incomplete or inaccurate.\n'
    'Use them as reference knowledge, not absolute rules. Check the current observation; missing facts are unknown.'
)


def library_fixture(value='on'):
    return {'schema_version': 'switch-cwm-library-v1', 'occurrences': [{
        'row': {'id': 'a', 'record_id': 'r', 'step_index': 0,
                'current_state_t': {'nodes': [], 'edges': []}},
        'action': {'operator': 'TURN_ON', 'participants': [
            {'role': 'target', 'node': 'kettle', 'node_type': 'appliance',
             'name': 'kettle', 'aliases': []}],
            'parameters': [], 'concepts': [], 'directions': [], 'owner_type': 'kettle'},
        'produced': [['kettle', 'power', value, 'supported_effect']],
        'provenance': {'source': 'synthetic'}}]}


def goal(key='power', value='ready'):
    return {'goal_facts': [{'object': 'kettle', 'key': key, 'value': value}]}


class SwitchContractTests(unittest.TestCase):
    def test_default_is_simple_and_strict_fields_only(self):
        graph = library_fixture()
        entry = graph['occurrences'][0]
        entry['row']['current_state_t']['nodes'] = [
            {'id': 'kettle', 'attributes': {'power': 'off'}}]
        entry['produced'] += [
            ['kettle', 'mode', 'boil', 'explicit_source_requirement'],
            ['kettle', 'temperature', 'warm', 'post_state_confirmation'],
            ['kettle', 'timer', 'long', 'typed_action_parameter']]
        library = SwitchCWM(graph)
        simple = library.retrieve('kettle', version='simple')
        self.assertEqual(library.retrieve('kettle'), simple)
        self.assertEqual(simple.context, STRICT_HEADER + '\n\nAction: TURN_ON(target=kettle)\n'
                         '  observed preconditions: kettle.power = "off"\n'
                         '  effects: kettle.power = "on"\n')
        node = simple.subgraph['nodes'][0]
        self.assertEqual(node['source_inventory'], {k: entry[k] for k in ('row', 'action', 'produced')})
        self.assertEqual(node['provenance'], entry['provenance'])
        self.assertFalse(node['execution_verified'])
        audit = simple.subgraph['audit']
        self.assertEqual((audit['version'], audit['additions']), ('simple', 'strict'))
        self.assertFalse(audit['local_graph_used'])
        self.assertFalse(audit['execution_verified'])
        self.assertEqual(audit['rendered_context_characters'], len(simple.context))
        json.dumps(simple.subgraph, allow_nan=False)

    def test_complex_preserves_rich_facts_and_duplicates(self):
        graph = library_fixture()
        entry = graph['occurrences'][0]
        entry['row']['current_state_t']['nodes'] = [
            {'id': 'kettle', 'attributes': {'power': 'off'}}]
        entry['action']['parameters'] = [{'attribute': 'timer', 'value': 'long'}]
        entry['produced'] += [
            ['kettle', 'mode', 'boil', 'explicit_source_requirement'],
            ['kettle', 'mode', 'boil', 'explicit_source_requirement'],
            ['kettle', 'temperature', 'warm', 'post_state_confirmation'],
            ['kettle', 'timer', 'long', 'typed_action_parameter'],
            ['kettle', 'heat', 'increased', 'derived_numeric_direction'],
            ['kettle', 'setting', 'chosen', 'typed_option_effect']]
        result = SwitchCWM(graph).retrieve('kettle', version='complex')
        self.assertTrue(result.context.startswith('TASK_CONDITIONED_CWM\nretrieval=task_text_tfidf_node_topk_'))
        self.assertIn('  requires:\n    - kettle.power=off', result.context)
        self.assertIn('  parameters:\n    - timer=long', result.context)
        for node, key, value, basis in entry['produced']:
            self.assertIn(f'    - {node}.{key}={value} basis={basis}', result.context)
        self.assertEqual(result.context.count('kettle.mode=boil basis=explicit_source_requirement'), 2)
        self.assertEqual(result.subgraph['nodes'][0]['source_inventory']['produced'], entry['produced'])
        self.assertEqual(result.subgraph['audit']['version'], 'complex')
        self.assertIsNone(result.subgraph['audit']['additions'])
        self.assertNotIn('DYNAMIC_LOCAL_GRAPH', result.context)
        json.dumps(result.subgraph, allow_nan=False)

    def test_actions_without_supported_effects(self):
        for produced in ([], [['kettle', 'mode', 'boil', 'explicit_source_requirement']]):
            for has_precondition in (False, True):
                with self.subTest(produced=produced, precondition=has_precondition):
                    graph = library_fixture()
                    entry = graph['occurrences'][0]
                    entry['produced'] = produced
                    if has_precondition:
                        entry['row']['current_state_t']['nodes'] = [
                            {'id': 'kettle', 'attributes': {'power': 'off'}}]
                    library = SwitchCWM(graph)
                    simple = library.retrieve('kettle')
                    detailed = library.retrieve('kettle', version='complex')
                    self.assertEqual([n['id'] for n in simple.subgraph['nodes']], ['a'] if has_precondition else [])
                    self.assertIn('effects: not specified', simple.context)
                    self.assertNotIn('boil', simple.context)
                    self.assertEqual([n['id'] for n in detailed.subgraph['nodes']], ['a'])
                    self.assertIn('ACTION_NODE action_1', detailed.context)
                    self.assertEqual(detailed.subgraph['nodes'][0]['source_inventory']['produced'], produced)
                    self.assertEqual(detailed.subgraph['audit']['hidden_no_strict_facts_ids'], [])

    def test_no_match_and_whole_action_budget(self):
        graph = library_fixture('x' * 1000)
        library = SwitchCWM(graph)
        for version in VERSIONS:
            with self.subTest(version=version):
                empty = library.retrieve(NO_MATCH, version=version)
                self.assertEqual(empty.subgraph['nodes'], [])
                self.assertEqual(empty.subgraph['edges'], [])
                self.assertEqual(empty.subgraph['audit']['omitted_ids'], [])
                self.assertNotIn('ACTION_NODE', empty.context)
                self.assertNotIn('OMITTED', empty.context)
                if version == 'simple':
                    self.assertIn('Action: no rules selected', empty.context)
                cap = len(empty.context) + 96
                result = library.retrieve('kettle', version=version, context_chars=cap)
                self.assertEqual(result.subgraph['nodes'], [])
                self.assertEqual(result.subgraph['audit']['omitted_ids'], ['a'])
                self.assertLessEqual(len(result.context), cap)
                self.assertNotIn('x' * 10, result.context)
                if version == 'complex':
                    self.assertTrue(result.context.endswith('[CWM_NODES_OMITTED_AT_FIXED_LIMIT=1]'))
                with self.assertRaisesRegex(ValueError, 'cap smaller'):
                    library.retrieve('kettle', version=version, context_chars=1)
                self.assertEqual(len(library.retrieve('kettle', version=version,
                                                     top_k=1, context_chars=14000).subgraph['nodes']), 1)

    def test_limits_require_positive_integers(self):
        library = SwitchCWM(library_fixture())
        for version in VERSIONS:
            for field in ('top_k', 'context_chars'):
                for value in (True, False, 0, -1, 1.5, '4', None):
                    with self.subTest(version=version, field=field, value=value):
                        with self.assertRaisesRegex(ValueError, 'positive integer'):
                            library.retrieve('kettle', version=version, **{field: value})
            with self.assertRaises(ValueError):
                library.retrieve('kettle', version=version, context_chars=14001)

    def test_strict_task_and_goal_inputs(self):
        library = SwitchCWM(library_fixture())
        for task in ('', ' \t ', None, 1):
            with self.subTest(task=task), self.assertRaises(ValueError):
                library.retrieve(task)
        for version in ('detailed', '', 1):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, 'version must be simple or complex'):
                library.retrieve('kettle', version=version)
        invalid_goals = [[], {'answer': 'A'}, {'goal_facts': [], 'answer': 'A'},
                         {'goal_facts': {}}, {'goal_facts': [{'object': 'kettle', 'key': 'power'}]},
                         {'goal_facts': [{'object': '', 'key': 'power', 'value': 'on'}]},
                         {'goal_facts': [{'object': 'kettle', 'key': 1, 'value': 'on'}]},
                         {'goal_facts': [{**goal()['goal_facts'][0], 'basis': 'answer'}]},
                         goal(value=float('nan'))]
        for value in invalid_goals:
            with self.subTest(goals=value), self.assertRaises(ValueError):
                library.retrieve('kettle', goals=value)
        for field in ('question', 'observations', 'options', 'answer'):
            with self.subTest(field=field), self.assertRaises(TypeError):
                library.retrieve('kettle', **{field: []})

    def test_source_control_characters_rejected_recursively(self):
        for char in ('\r', '\n', '\x00'):
            for field in ('produced', 'alias', 'state_key'):
                with self.subTest(char=repr(char), field=field):
                    graph = library_fixture()
                    entry = graph['occurrences'][0]
                    if field == 'produced':
                        entry['produced'][0][2] = 'off' + char + 'injected'
                    elif field == 'alias':
                        entry['action']['participants'][0]['aliases'] = ['kettle' + char]
                    else:
                        entry['row']['current_state_t']['nodes'] = [
                            {'id': 'kettle', 'attributes': {'power' + char: 'off'}}]
                    with self.assertRaisesRegex(ValueError, 'CR, LF, or NUL'):
                        SwitchCWM(graph)

    def test_explicit_goals_cache_hit_and_cache_miss(self):
        graph = library_fixture('ready')
        graph['task_goals'] = {NO_MATCH: goal()}
        library = SwitchCWM(graph)
        for version in VERSIONS:
            with self.subTest(version=version):
                cached = library.retrieve(NO_MATCH, version=version)
                self.assertEqual(cached.subgraph['audit']['goal_source'], 'bundled_task_cache')
                self.assertEqual(cached.subgraph['audit']['fuzzy_ids'], [])
                self.assertEqual([n['id'] for n in cached.subgraph['nodes']], ['a'])
                explicit = library.retrieve(NO_MATCH, version=version, goals=goal())
                self.assertEqual(explicit.context, cached.context)
                self.assertEqual(explicit.subgraph['audit']['goal_source'], 'explicit_task_goals')
                overridden = library.retrieve(NO_MATCH, version=version, goals={'goal_facts': []})
                self.assertEqual(overridden.subgraph['nodes'], [])
                miss = library.retrieve('kettle newly worded task', version=version)
                self.assertEqual(miss.subgraph['audit']['goal_source'], 'absent')
                self.assertEqual(miss.subgraph['audit']['goal_facts'], [])
                self.assertEqual([n['id'] for n in miss.subgraph['nodes']], ['a'])

    def test_nested_goal_and_dependency_witness_mutations_are_isolated(self):
        graph = library_fixture('ready')
        successor = copy.deepcopy(graph['occurrences'][0])
        successor['row'].update(id='b', step_index=1)
        successor['row']['current_state_t']['nodes'] = [
            {'id': 'kettle', 'attributes': {'power': 'ready'}}]
        successor['produced'] = [['kettle', 'done', 'yes', 'supported_effect']]
        graph['occurrences'].append(successor)
        library = SwitchCWM(graph)
        goals = goal('done', 'yes')
        original_goals = copy.deepcopy(goals)
        for version in VERSIONS:
            with self.subTest(version=version):
                result = library.retrieve('kettle', version=version, goals=goals)
                expected = copy.deepcopy(result)
                audit = result.subgraph['audit']
                self.assertEqual({(e['source'], e['target']) for e in result.subgraph['edges']}, {('a', 'b')})
                audit['goal_witnesses']['b'][0]['witnesses'][0][2] = 'CORRUPTED'
                audit['goal_witnesses']['b'][0]['goal']['value'] = 'CORRUPTED'
                audit['goal_facts'][0]['value'] = 'CORRUPTED'
                audit['dependency_links'][0]['witnesses'][0][2] = 'CORRUPTED'
                result.subgraph['edges'][0]['witnesses'][0][2] = 'CORRUPTED'
                result.subgraph['nodes'][0]['source_inventory']['produced'][0][2] = 'CORRUPTED'
                self.assertEqual(library.retrieve('kettle', version=version, goals=goals), expected)
                self.assertEqual(goals, original_goals)
                self.assertNotIn('CORRUPTED', json.dumps(graph))

    def test_removed_additions_keyword_is_rejected(self):
        with self.assertRaises(TypeError):
            SwitchCWM(library_fixture(), additions='strict')
        with self.assertRaises(TypeError):
            SwitchCWM.load(additions='strict')
        with self.assertRaises(TypeError):
            SwitchCWM(library_fixture()).retrieve('kettle', additions='strict')


class SwitchIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / 'data' / 'switch' / 'causground.json.gz'
        cls.direct = SwitchCWM.load(cls.path)
        cls.shared = CausGround.load(cls.path)

    def test_shared_api_matches_entire_direct_result(self):
        self.assertEqual(self.shared.profile, 'switch')
        task = 'Turn on the kettle.'
        for version in (None, *VERSIONS):
            with self.subTest(version=version):
                result = self.shared.retrieve(task, version=version)
                direct = self.direct.retrieve(task, version=version)
                self.assertEqual(result.context, direct.context)
                self.assertEqual(result.subgraph, direct.subgraph)
        for kwargs in ({'mode': 'causal'}, {'state': {'objects': []}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.shared.retrieve(task, **kwargs)

    def test_shared_and_standalone_cli_stdout(self):
        task = 'Turn on the kettle.'
        for main, prefix in ((shared_main, ['--env', 'switch']), (standalone_main, ['query'])):
            for version in (None, *VERSIONS):
                for structured in (False, True):
                    with self.subTest(cli=main.__module__, version=version, structured=structured):
                        args = [*prefix, '--task', task]
                        if version is not None:
                            args += ['--version', version]
                        if structured:
                            args += ['--json']
                        stdout, stderr = io.StringIO(), io.StringIO()
                        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                            code = main(args)
                        self.assertEqual(code, 0, stderr.getvalue())
                        self.assertEqual(stderr.getvalue(), '')
                        direct = self.direct.retrieve(task, version=version)
                        if structured:
                            self.assertEqual(json.loads(stdout.getvalue()),
                                             json.loads(json.dumps(direct.subgraph, allow_nan=False)))
                        else:
                            suffix = '\n' if main is shared_main else ''
                            self.assertEqual(stdout.getvalue(), direct.context + suffix)

    def test_removed_additions_cli_is_argparse_error(self):
        for main, prefix in ((shared_main, ['--env', 'switch']), (standalone_main, ['query'])):
            with self.subTest(cli=main.__module__):
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        main([*prefix, '--task', 'kettle', '--additions', 'strict'])
                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(stdout.getvalue(), '')
                self.assertIn('--additions', stderr.getvalue())


if __name__ == '__main__':
    unittest.main(verbosity=2)
