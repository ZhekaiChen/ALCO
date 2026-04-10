# Prompt Template Specification

This file defines the required prompt contract for action-level TSP rollout.

## Prompt goals

The model must:
1. understand the full TSP instance,
2. reason globally about the whole tour,
3. choose exactly one next node given the current partial route,
4. output detailed reasoning text,
5. end with exactly one machine-readable final tag.

## Required final tag

The response must contain exactly one final tag of the form:

`<FINAL_NEXT_NODE>INTEGER</FINAL_NEXT_NODE>`

This is the only part of the response that the parser should use for action extraction.

## Required prompt fields

A rollout prompt should include, in order:

1. A task description stating that the model must choose the next node in a step-by-step TSP solution process.
2. The number of nodes.
3. A list of nodes and 2D coordinates.
4. The current partial route as an ordered prefix.
5. The visited set and the remaining unvisited nodes, if helpful.
6. The current node / position, if included by configuration.
7. An instruction to reason globally, not greedily.
8. An instruction to provide detailed reasoning.
9. A strict final-output instruction requiring exactly one final tag.

## Suggested wording

You are solving a Traveling Salesman Problem step by step.
Your job in this step is not to output the full tour, but to choose exactly one next node to append to the current partial route.
Think carefully about the global tour quality before deciding.
Provide detailed reasoning about the overall route structure and the tradeoffs among candidate next nodes.
At the end, output exactly one final answer in the format:
<FINAL_NEXT_NODE>INTEGER</FINAL_NEXT_NODE>
Do not output multiple final tags.

## Parsing rule

- The parser ignores free-form reasoning content.
- The parser extracts only the integer inside the final tag.
- Missing, duplicated, or malformed final tags are parsing failures.

## Logging rule

The full prompt and full model output must be stored in rollout logs.
