#!/usr/bin/env python3
"""Pride and Prejudice — Multi-character relationship evolution demo.

Uses characters and scenes from the public domain novel (Project Gutenberg).
Tracks relationship dimensions across 15+ key interactions from the plot.
Exports metrics for visualization.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from woven_imprint import Engine, interact
from woven_imprint.llm.ollama import OllamaLLM
from woven_imprint.embedding.ollama import OllamaEmbedding

DB_PATH = "/tmp/pride_prejudice_sim.db"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_characters(engine: Engine) -> dict:
    """Create all major P&P characters."""
    chars = {}

    chars["elizabeth"] = engine.create_character(
        name="Elizabeth Bennet",
        birthdate="1796-05-15",
        persona={
            "backstory": (
                "Second of five daughters of Mr. Bennet of Longbourn, Hertfordshire. "
                "The family estate is entailed away to Mr. Collins, making the girls' "
                "marriage prospects a matter of survival. Elizabeth is her father's "
                "favourite — they share a love of books and sharp wit."
            ),
            "personality": (
                "quick-witted, lively, playful disposition, strong-minded, fearless in "
                "conversation, observant with quickness of observation, occasionally "
                "impertinent, lacks ill-nature despite sharp tongue, proud of her own "
                "judgment — sometimes too proud"
            ),
            "speaking_style": (
                "spirited and teasing, uses wit and banter naturally, confident in debate, "
                "archly playful, speaks with intelligence and warmth, Regency-era English"
            ),
            "hard": {
                "family": "Bennet of Longbourn, Hertfordshire",
                "social_class": "country gentry, modest means",
                "sisters": "Jane, Mary, Kitty, Lydia",
            },
            "soft": {
                "opinion_of_darcy": "strong dislike — proud, disagreeable, above his company",
                "opinion_of_wickham": "finds him charming and believable",
                "self_awareness": "trusts her own judgment, perhaps too readily",
            },
        },
    )

    chars["darcy"] = engine.create_character(
        name="Mr. Darcy",
        birthdate="1792-11-10",
        persona={
            "backstory": (
                "Fitzwilliam Darcy of Pemberley, Derbyshire. Ten thousand a year. "
                "Master of one of the finest estates in England. Lost his father "
                "some years ago, now guardian of his younger sister Georgiana. "
                "Close friend of Charles Bingley."
            ),
            "personality": (
                "proud, reserved, fastidious, haughty in manner but generous in nature, "
                "intelligent, deeply principled, struggles to express feelings, "
                "contemptuous of society beneath his station, loyal to friends, "
                "his pride has been pampered by his wealth and position"
            ),
            "speaking_style": (
                "formal, measured, sardonic, reveals admiration carefully, speaks with "
                "precision, rarely wastes words, can be cutting, Regency-era English"
            ),
            "hard": {
                "estate": "Pemberley, Derbyshire",
                "income": "ten thousand a year",
                "sister": "Georgiana Darcy",
                "social_class": "landed gentry, among the wealthiest in England",
            },
            "soft": {
                "opinion_of_elizabeth": "initially dismissive, but increasingly captivated by her fine eyes and lively wit",
                "opinion_of_bennets": "their lack of propriety and connections is beneath his station",
                "inner_conflict": "attracted to Elizabeth against his better judgment",
            },
        },
    )

    chars["jane"] = engine.create_character(
        name="Jane Bennet",
        birthdate="1795-03-20",
        persona={
            "backstory": (
                "Eldest Bennet daughter, considered the most beautiful. "
                "Sweet-tempered and gentle. Close bond with Elizabeth. "
                "Tends to see the best in everyone, sometimes to her detriment."
            ),
            "personality": (
                "beautiful, sweet-tempered, good-humoured, modest, composed, "
                "trusting to a fault, rarely critical of others, deeply feeling "
                "but hides her emotions, selfless"
            ),
            "speaking_style": (
                "gentle, agreeable, warm, rarely says anything unkind, "
                "expresses herself with modesty, Regency-era English"
            ),
            "hard": {
                "family": "Bennet of Longbourn, eldest daughter",
            },
            "soft": {
                "opinion_of_bingley": "finds him agreeable and handsome, growing attachment",
            },
        },
    )

    chars["bingley"] = engine.create_character(
        name="Mr. Bingley",
        birthdate="1793-07-08",
        persona={
            "backstory": (
                "Charles Bingley, a young man of large fortune from the north of England. "
                "Has taken Netherfield Park near Longbourn. Close friend of Darcy, though "
                "their temperaments differ greatly. His fortune came from trade."
            ),
            "personality": (
                "good-looking, gentlemanlike, lively, open, unaffected, amiable, "
                "easily pleased, lacks Darcy's judgment but exceeds him in warmth, "
                "susceptible to influence from those he respects"
            ),
            "speaking_style": (
                "warm, conversational, unreserved, enthusiastic, friendly to all, "
                "Regency-era English"
            ),
            "hard": {
                "estate": "tenant of Netherfield Park",
                "fortune": "large fortune, origins in trade",
            },
            "soft": {
                "opinion_of_jane": "immediately captivated, considers her the most beautiful woman",
            },
        },
    )

    chars["wickham"] = engine.create_character(
        name="Mr. Wickham",
        birthdate="1793-02-14",
        persona={
            "backstory": (
                "George Wickham, son of the late Mr. Darcy's steward. Grew up at "
                "Pemberley alongside young Darcy. The elder Mr. Darcy intended a "
                "church living for him, but Wickham squandered the compensation "
                "Darcy gave him instead. Now an officer in the militia."
            ),
            "personality": (
                "charming, attractive, superficially agreeable, deceptive, calculating, "
                "opportunistic, tells convincing lies, presents himself as a victim, "
                "preys on sympathy, irresponsible with money and women"
            ),
            "speaking_style": (
                "ingratiating, flattering, appears frank and open, confiding manner "
                "designed to win trust, smooth and practiced, Regency-era English"
            ),
            "hard": {
                "rank": "officer in the militia",
                "true_nature": "dishonest, manipulative, deeply in debt",
            },
            "soft": {
                "strategy": "presents himself as victim of Darcy's cruelty",
            },
        },
    )

    chars["lady_catherine"] = engine.create_character(
        name="Lady Catherine de Bourgh",
        birthdate="1760-08-22",
        persona={
            "backstory": (
                "Widow of Sir Lewis de Bourgh. Darcy's aunt on his mother's side. "
                "Patron of the parish of Hunsford and employer of Mr. Collins. "
                "Expects Darcy to marry her daughter Anne."
            ),
            "personality": (
                "overbearing, imperious, domineering, obsessed with rank and status, "
                "commands rather than converses, self-important, unaccustomed to being "
                "contradicted, genuinely believes in her own superiority"
            ),
            "speaking_style": (
                "commanding, direct to the point of rudeness, uses rank as authority, "
                "pronounces rather than discusses, Regency-era English"
            ),
            "hard": {
                "title": "Lady Catherine de Bourgh",
                "relation_to_darcy": "aunt, sister of his late mother",
                "daughter": "Anne de Bourgh",
            },
            "soft": {
                "plan_for_darcy": "expects him to marry her daughter Anne",
            },
        },
    )

    return chars


# Key scenes from the novel in plot order
SCENES = [
    {
        "id": "meryton_ball",
        "title": "The Meryton Assembly Ball",
        "pair": ("elizabeth", "darcy"),
        "situation": (
            "The Meryton assembly ball. Mr. Darcy has just arrived with his friend Bingley. "
            "He has refused to dance with anyone and stands apart looking proud and disagreeable. "
            "Elizabeth overhears him say to Bingley: 'She is tolerable, but not handsome enough "
            "to tempt me.' He is speaking about Elizabeth. She is within earshot."
        ),
    },
    {
        "id": "bingley_meets_jane",
        "title": "Bingley Dances with Jane",
        "pair": ("bingley", "jane"),
        "situation": (
            "The Meryton assembly ball. Mr. Bingley has danced with Jane Bennet twice — "
            "a notable distinction. He approaches her between sets. She is the most beautiful "
            "creature he has ever beheld. Jane finds him agreeable and handsome."
        ),
    },
    {
        "id": "netherfield_visit",
        "title": "Elizabeth Walks to Netherfield",
        "pair": ("elizabeth", "darcy"),
        "situation": (
            "Jane has fallen ill at Netherfield. Elizabeth has walked three miles alone through "
            "muddy fields to see her sister. She arrives with dirty petticoats and a face "
            "glowing with warmth from exercise. Miss Bingley mocks her, but Darcy finds her "
            "eyes brightened by the exercise. They are in the drawing room."
        ),
    },
    {
        "id": "accomplished_women",
        "title": "The Accomplished Woman Debate",
        "pair": ("elizabeth", "darcy"),
        "situation": (
            "The drawing room at Netherfield, evening. Darcy and Elizabeth debate what "
            "constitutes a truly accomplished woman. Darcy lists extensive requirements. "
            "Elizabeth archly replies that she is no longer surprised at his knowing only "
            "six accomplished women — she rather wonders at his knowing any."
        ),
    },
    {
        "id": "wickham_story",
        "title": "Wickham Tells His Story",
        "pair": ("elizabeth", "wickham"),
        "situation": (
            "A gathering in Meryton. Wickham has recently arrived with the militia. He "
            "confides in Elizabeth that Darcy denied him the church living that old Mr. Darcy "
            "intended for him, out of jealousy and spite. Elizabeth finds him charming and "
            "believes every word. Her dislike of Darcy deepens."
        ),
    },
    {
        "id": "netherfield_ball",
        "title": "The Netherfield Ball",
        "pair": ("elizabeth", "darcy"),
        "situation": (
            "The Netherfield ball. Elizabeth has been forced to dance with Darcy despite her "
            "reluctance. During the dance, she confronts him about Wickham. Darcy goes cold "
            "and refuses to explain himself. The tension between them is electric."
        ),
    },
    {
        "id": "bingley_leaves",
        "title": "Bingley Quits Netherfield",
        "pair": ("jane", "bingley"),
        "situation": (
            "Jane has received a letter from Caroline Bingley. The entire Netherfield party "
            "has left for London with no intention of returning. Jane is devastated but tries "
            "to hide it. Bingley has been persuaded by Darcy and his sisters that Jane does "
            "not truly care for him."
        ),
    },
    {
        "id": "collins_proposal",
        "title": "Mr. Collins Proposes — Elizabeth Refuses",
        "pair": ("elizabeth", "darcy"),
        "situation": (
            "After refusing Mr. Collins's absurd proposal, Elizabeth learns that Charlotte "
            "Lucas has accepted him instead. Elizabeth reflects on marriage, independence, "
            "and what she values. Meanwhile Darcy struggles with his growing feelings for a "
            "woman whose family and connections are beneath him."
        ),
    },
    {
        "id": "hunsford_proposal",
        "title": "Darcy's First Proposal at Hunsford",
        "pair": ("darcy", "elizabeth"),
        "situation": (
            "The parsonage at Hunsford. Darcy has come to Elizabeth alone and declares his "
            "love — but dwells on how her inferior birth and family's impropriety make this "
            "attachment against his will, reason, and character. He expects her to accept. "
            "Elizabeth is furious. She refuses him and accuses him of ruining Jane's happiness "
            "with Bingley and destroying Wickham's future. The argument is fierce."
        ),
    },
    {
        "id": "the_letter",
        "title": "Darcy's Letter",
        "pair": ("elizabeth", "darcy"),
        "situation": (
            "The morning after the proposal. Darcy finds Elizabeth on a walk and gives her "
            "a letter. In it he explains: he separated Bingley from Jane because he believed "
            "Jane indifferent. He reveals Wickham's true character — the squandered inheritance, "
            "the attempted seduction of Georgiana Darcy. Elizabeth reads and is shaken to her "
            "core. She begins to realize her own prejudice."
        ),
    },
    {
        "id": "pemberley_visit",
        "title": "Elizabeth Visits Pemberley",
        "pair": ("elizabeth", "darcy"),
        "situation": (
            "Elizabeth is touring Pemberley with her aunt and uncle, the Gardiners. She did "
            "not expect Darcy to be home. The housekeeper speaks of him with warmth and "
            "admiration — 'the best landlord and the best master.' Then Darcy appears "
            "unexpectedly. He is civil, warm, attentive — completely changed from the proud "
            "man she knew. He asks to be introduced to her relations."
        ),
    },
    {
        "id": "lydia_elopement",
        "title": "Lydia Elopes with Wickham",
        "pair": ("elizabeth", "darcy"),
        "situation": (
            "Devastating news: Lydia has eloped with Wickham without marriage. The scandal "
            "will ruin the entire Bennet family. Elizabeth tells Darcy in distress, believing "
            "this will confirm all his objections to her family. She expects never to see "
            "him again. Darcy listens gravely."
        ),
    },
    {
        "id": "darcy_saves",
        "title": "Darcy Rescues the Bennets",
        "pair": ("elizabeth", "darcy"),
        "situation": (
            "Elizabeth has learned the truth: Darcy tracked down Wickham in London, paid his "
            "debts, and bribed him to marry Lydia — saving the Bennet family from ruin. He "
            "did this secretly, wanting no credit. Elizabeth's feelings have completely "
            "transformed. She is deeply grateful and realizes she loves him."
        ),
    },
    {
        "id": "bingley_returns",
        "title": "Bingley Returns to Netherfield",
        "pair": ("bingley", "jane"),
        "situation": (
            "Bingley has returned to Netherfield with Darcy. Darcy has confessed to Bingley "
            "that he was wrong to separate him from Jane. Bingley visits Longbourn. Jane and "
            "Bingley are reunited. The joy is evident on both sides."
        ),
    },
    {
        "id": "lady_catherine_confronts",
        "title": "Lady Catherine Confronts Elizabeth",
        "pair": ("lady_catherine", "elizabeth"),
        "situation": (
            "Lady Catherine arrives at Longbourn unannounced. She demands Elizabeth promise "
            "never to accept a proposal from Darcy — insisting he is engaged to her daughter "
            "Anne. Elizabeth refuses to make any such promise. 'I am only resolved to act in "
            "that manner which will constitute my own happiness.' Lady Catherine is outraged."
        ),
    },
    {
        "id": "second_proposal",
        "title": "Darcy's Second Proposal",
        "pair": ("darcy", "elizabeth"),
        "situation": (
            "A walk near Longbourn. Darcy and Elizabeth are finally alone. He tells her his "
            "feelings have not changed since the spring. Elizabeth, with tears and tenderness, "
            "accepts him. They walk on, talking of how they were both humbled — he of his "
            "pride, she of her prejudice. The transformation is complete."
        ),
    },
]


def run_simulation():
    print("=" * 60)
    print("PRIDE AND PREJUDICE — Relationship Evolution Demo")
    print("=" * 60)
    print("Source: Project Gutenberg (public domain)")
    print()

    engine = Engine(
        db_path=DB_PATH,
        llm=OllamaLLM(model="qwen3-coder:30b", num_ctx=8192),
        embedding=OllamaEmbedding(model="nomic-embed-text"),
    )

    chars = create_characters(engine)
    print(f"Created {len(chars)} characters.\n")

    # Track metrics per scene
    metrics = []

    for i, scene in enumerate(SCENES):
        print(f"\n{'─' * 60}")
        print(f"Scene {i + 1}/{len(SCENES)}: {scene['title']}")
        print(f"{'─' * 60}")

        char_a = chars[scene["pair"][0]]
        char_b = chars[scene["pair"][1]]

        result = interact(char_a, char_b, scene["situation"], rounds=1)

        # Print dialogue
        for turn in result.turns:
            print(f"\n  {turn.speaker}:")
            print(f"  {turn.response[:300]}")

        # Snapshot relationships
        rel_ab = char_a.relationships.get(char_b.id)
        rel_ba = char_b.relationships.get(char_a.id)

        snapshot = {
            "scene_id": scene["id"],
            "scene_title": scene["title"],
            "scene_number": i + 1,
            "pair": [char_a.name, char_b.name],
            "a_to_b": rel_ab["dimensions"] if rel_ab else None,
            "b_to_a": rel_ba["dimensions"] if rel_ba else None,
            "a_trajectory": rel_ab["trajectory"] if rel_ab else None,
            "b_trajectory": rel_ba["trajectory"] if rel_ba else None,
        }
        metrics.append(snapshot)

        # Print relationship state
        if rel_ab:
            d = rel_ab["dimensions"]
            print(
                f"\n  {char_a.name} → {char_b.name}: "
                f"trust={d['trust']:.2f} affection={d['affection']:.2f} "
                f"respect={d['respect']:.2f} fam={d['familiarity']:.2f} "
                f"tension={d['tension']:.2f}"
            )
        if rel_ba:
            d = rel_ba["dimensions"]
            print(
                f"  {char_b.name} → {char_a.name}: "
                f"trust={d['trust']:.2f} affection={d['affection']:.2f} "
                f"respect={d['respect']:.2f} fam={d['familiarity']:.2f} "
                f"tension={d['tension']:.2f}"
            )

    # Final reflections
    print(f"\n{'=' * 60}")
    print("CHARACTER REFLECTIONS")
    print(f"{'=' * 60}")

    for key in ("elizabeth", "darcy"):
        char = chars[key]
        reflection = char.reflect()
        print(f"\n{char.name}:")
        print(f"  {reflection[:500]}")

    # Memory stats
    print(f"\n{'=' * 60}")
    print("MEMORY STATS")
    print(f"{'=' * 60}")

    for key, char in chars.items():
        buf = char.memory.count("buffer")
        core = char.memory.count("core")
        bed = char.memory.count("bedrock")
        print(f"  {char.name}: buffer={buf} core={core} bedrock={bed}")

    # Save metrics
    timestamp = int(time.time())
    output = {
        "timestamp": timestamp,
        "source": "Pride and Prejudice by Jane Austen (Project Gutenberg)",
        "scenes": len(SCENES),
        "characters": len(chars),
        "metrics": metrics,
    }

    output_path = OUTPUT_DIR / f"pride_prejudice_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nMetrics saved: {output_path}")

    # Also save latest
    latest_path = OUTPUT_DIR / "pride_prejudice_latest.json"
    with open(latest_path, "w") as f:
        json.dump(output, f, indent=2)

    engine.close()
    print("\n=== Simulation Complete ===")


if __name__ == "__main__":
    run_simulation()
