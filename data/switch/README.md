# Bundled SWITCH source library

`causground.json.gz` contains 3,119 normalized source occurrences and 564 exact-task goal-cache entries, using schema `switch-cwm-library-v1`. `tasks.json` provides example tasks.

Each occurrence contains `row`, `action`, `produced`, and `provenance`. Produced facts use `[node, attribute, value, basis]`. Goal-cache entries map task text to `{goal_facts: [{object, key, value}]}`.

Retrieval uses task text only. No evaluation questions, answer options, answer labels, media, or local observation graphs are included. Uncached tasks still use lexical retrieval and source dependencies.

`simple` shows Action, observed preconditions, and supported effects. `complex` retains the original detailed action blocks. Both return the source inventory and dependency evidence in `subgraph.json`. Source facts are trajectory evidence, not guaranteed causal laws; object IDs are local to each source record.

The upstream SWITCH-Basic data card declares CC BY-NC 4.0. See [source and license notices](../../THIRD_PARTY_NOTICES.md).
