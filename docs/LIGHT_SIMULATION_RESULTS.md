# LIGHT World Simulation — Results

> All character data sourced from the [LIGHT dataset](https://parl.ai/projects/light/) (Facebook Research, MIT License).
> 663 locations, 3462 objects, 1755 characters described entirely in natural language.

## Overview

| Metric | Value |
|--------|-------|
| Total LIGHT characters | 1324 |
| Total LIGHT rooms | 661 |
| Active characters (in rooms with 3+) | 282 |
| Active rooms | 107 |
| Idle characters (visible, no interaction) | 1042 |
| Simulation turns | 5 |

## Turn-by-Turn Evolution

| Turn | Events | Relationships | Positive | Negative | Neutral | Avg Trust | Avg Tension | Avg Familiarity |
|------|--------|---------------|----------|----------|---------|-----------|-------------|-----------------|
| 1 | 282 | 175 | 0 | 1 | 174 | 0.018 | 0.009 | 0.029 |
| 2 | 282 | 175 | 0 | 7 | 168 | 0.036 | 0.017 | 0.056 |
| 3 | 282 | 175 | 3 | 10 | 162 | 0.054 | 0.024 | 0.079 |
| 4 | 282 | 175 | 46 | 10 | 119 | 0.072 | 0.03 | 0.105 |
| 5 | 282 | 175 | 132 | 11 | 32 | 0.093 | 0.04 | 0.131 |

## Top 20 Relationships by Trust

| Character A | Character B | Trust | Affection | Tension | Familiarity | Type |
|-------------|-------------|-------|-----------|---------|-------------|------|
| repentant person | waiting priest | 0.21 | 0.13 | 0.0 | 0.28 | stranger |
| cook | hunters | 0.18 | 0.13 | 0.0 | 0.16 | stranger |
| family | Master craftsman | 0.18 | 0.15 | 0.0 | 0.28 | stranger |
| princes | butlers | 0.17 | 0.12 | 0.0 | 0.12 | stranger |
| young woman | younger children | 0.17 | 0.14 | 0.0 | 0.3 | stranger |
| people saved by the Paladins | local merchants | 0.17 | 0.12 | 0.0 | 0.21 | stranger |
| Close alliances | family | 0.17 | 0.12 | 0.0 | 0.2 | stranger |
| merpeoples | surface dwellers | 0.17 | 0.11 | 0.0 | 0.34 | stranger |
| surface dwellers | Merchants | 0.17 | 0.12 | 0.0 | 0.15 | stranger |
| pligrims | cardinal | 0.16 | 0.13 | 0.0 | 0.25 | stranger |
| previous queens of other eras | Future Heir | 0.16 | 0.1 | 0.0 | 0.12 | stranger |
| men | troops | 0.16 | 0.09 | 0.0 | 0.09 | stranger |
| well off business man | wealthy bookshop owner | 0.16 | 0.15 | 0.0 | 0.07 | stranger |
| sons | daughter | 0.16 | 0.13 | 0.0 | 0.24 | stranger |
| sword-maker | blacksmith | 0.16 | 0.11 | 0.0 | 0.18 | stranger |
| priest | citizens of | 0.15 | 0.1 | 0.0 | 0.12 | stranger |
| father | son | 0.15 | 0.16 | 0.0 | 0.12 | stranger |
| churchgoer | repentant person | 0.15 | 0.1 | 0.0 | 0.05 | stranger |
| their family | 3 servants | 0.15 | 0.1 | 0.0 | 0.19 | stranger |
| kings | queens | 0.15 | 0.09 | 0.0 | 0.06 | stranger |

## Most Tense Relationships

| Character A | Character B | Tension | Trust |
|-------------|-------------|---------|-------|
| A powerful sorcerer | child-like village people | 0.52 | -0.38 |
| torture master | bloodied prisoner | 0.51 | -0.32 |
| their enemies | The torture master | 0.48 | -0.29 |
| noble | fierce assassin | 0.47 | -0.2 |
| child-like peoples | master wizard | 0.46 | -0.2 |
| laborsmen | poorer subsistence farmers | 0.43 | -0.3 |
| master wizard | A powerful sorcerer | 0.43 | -0.26 |
| man | Family | 0.41 | -0.13 |
| enemies | torture master | 0.32 | -0.18 |
| Royal family members | artists | 0.31 | -0.08 |

## Biggest Relationship Changes (Trust + Tension Delta)

| Character A | Character B | Total Delta | Trust History |
|-------------|-------------|-------------|---------------|
| A powerful sorcerer | child-like village people | 0.77 | -0.08 → -0.15000000000000002 → -0.23000000000000004 → -0.31000000000000005 → -0.38000000000000006 |
| their enemies | The torture master | 0.66 | -0.04 → -0.11000000000000001 → -0.19 → -0.26 → -0.29000000000000004 |
| child-like peoples | master wizard | 0.64 | 0.02 → 0.03 → -0.09 → -0.16 → -0.2 |
| laborsmen | poorer subsistence farmers | 0.61 | -0.05 → -0.12000000000000001 → -0.2 → -0.25 → -0.3 |
| torture master | bloodied prisoner | 0.56 | -0.12 → -0.15 → -0.18 → -0.25 → -0.32 |
| master wizard | A powerful sorcerer | 0.52 | -0.08 → -0.13 → -0.21000000000000002 → -0.29000000000000004 → -0.26 |
| noble | fierce assassin | 0.47 | -0.08 → -0.2 → -0.18000000000000002 → -0.16000000000000003 → -0.20000000000000004 |
| bloodied prisoner | monarchs | 0.44 | 0.02 → -0.009999999999999998 → -0.08 → -0.1 → -0.13 |
| enemies | torture master | 0.43 | -0.03 → -0.08 → -0.15000000000000002 → -0.13000000000000003 → -0.18000000000000005 |
| man | Family | 0.37 | -0.05 → -0.08 → -0.11 → -0.15 → -0.13 |
| Recently tortured | those who have committed crimes | 0.31 | 0.02 → 0.03 → 0.03 → -0.05 → -0.08 |
| Royal family members | artists | 0.31 | -0.03 → -0.05 → -0.12000000000000001 → -0.1 → -0.08 |
| torturer | murderer | 0.29 | 0.03 → -0.010000000000000002 → 0.009999999999999998 → -0.02 → -0.07 |
| King | castle guard | 0.29 | -0.05 → -0.13 → -0.18 → -0.18 → -0.16 |
| royals | rabid dogs | 0.25 | 0.02 → 0.05 → 0.07 → 0.09000000000000001 → 0.11000000000000001 |

## Most Active Rooms

| Room | Category | Characters | Total Events |
|------|----------|------------|--------------|
| Rectory | Inside Church | 7 | 35 |
| Torture Room | Dungeon | 6 | 30 |
| Yurt | Desert | 5 | 25 |
| arrow house | Outside Castle | 5 | 25 |
| Castle Kitchen | Inside Castle | 5 | 25 |
| Outpost | Mountain | 5 | 25 |
| Town | Town | 5 | 25 |
| Black Smiths shop | Town | 5 | 25 |
| Hillside Manor | Countryside | 5 | 25 |
| The grand dining room | Inside Palace | 4 | 20 |
| The Troll's Lair | Cave | 4 | 20 |
| Beyond the Wall | Outside Castle | 4 | 20 |
| Dungeon | Inside Castle | 4 | 20 |
| The werewolves tavern | Tavern | 4 | 20 |
| Royal Tomb | Inside Temple | 4 | 20 |
| Training Fields | Outside Palace | 4 | 20 |
| Confessional Room | Inside Church | 4 | 20 |
| The Queen's Chamber | Inside Palace | 4 | 20 |
| The throne room | Inside Palace | 4 | 20 |
| A wooded magical village | Forest | 4 | 20 |

## Category Breakdown

| Category | Rooms | Characters |
|----------|-------|------------|
| Inside Palace | 15 | 41 |
| Inside Church | 7 | 24 |
| Inside Castle | 7 | 18 |
| Port | 8 | 17 |
| Town | 4 | 15 |
| Inside Tower | 8 | 15 |
| Dungeon | 5 | 15 |
| Inside Temple | 5 | 14 |
| Bazaar | 6 | 12 |
| Outside Castle | 3 | 11 |
| Tavern | 3 | 9 |
| Outside Palace | 5 | 9 |
| Desert | 2 | 8 |
| Countryside | 2 | 8 |
| Forest | 3 | 7 |
| Mountain | 2 | 7 |
| underwater aquapolis | 2 | 6 |
| city in the clouds | 2 | 6 |
| Inside Cottage | 3 | 5 |
| Outside Church | 2 | 4 |
| Cave | 1 | 4 |
| supernatural | 1 | 4 |
| Shore | 2 | 3 |
| Wasteland | 1 | 3 |
| Swamp | 1 | 3 |
| Lake | 2 | 3 |
| Outside Tower | 1 | 3 |
| netherworld | 1 | 3 |
| Outside Temple | 1 | 2 |
| Abandoned | 1 | 2 |
| Outside Cottage | 1 | 1 |

---
*Generated by [Woven Theatre](https://github.com/virtaava/woven-theatre) using [woven-imprint](https://github.com/virtaava/woven-imprint) v0.4.0*