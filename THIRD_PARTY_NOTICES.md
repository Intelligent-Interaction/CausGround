# Source and license notices

This extension was prepared for the Intelligent-Interaction/CausGround maintainers against public revision `2ecf27a90aac25b31ccb520c644927099012ee39` (verified 2026-11-17):
https://github.com/Intelligent-Interaction/CausGround

The public revision did not declare a repository code license. This package does not invent or grant a new code license. The repository maintainers should state their chosen code license when publishing. The shared API and query command retain the original environment implementations and add explicit SWITCH dispatch.

The upstream SWITCH-Basic v1 data card declares **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**:
https://creativecommons.org/licenses/by-nc/4.0/

SWITCH: Benchmarking Modeling and Handling of Tangible Interfaces in Long-horizon Embodied Scenarios. Jieru Lin, Zhiwei Yu, Börje F. Karlsson (2025). Upstream project/leaderboard:
https://huggingface.co/spaces/BAAI-Agents/SwitchBasic-Leaderboard

`data/switch/causground.json.gz` contains normalized trajectory-derived knowledge and task-only goal facts from the historical CWM preparation, including later source-supported action/requirement compilations. It is a derived resource, not the upstream benchmark distribution. Occurrence IDs, source-record IDs, and source-step indices are retained for provenance. This package does not establish broader rights to nonpublic source annotations or original media. The data card's noncommercial restriction must not be represented as a permissive software license or an independently verified license for every historical input.

No benchmark questions, answer options, answer labels, evaluation media, inference runners, private credentials, or server filesystem paths are included in the distributable resource. The goal cache contains task text only and is explicitly separate from answer labels. Source facts retain their original evidence basis; intended facts are not evidence of observed success.
