Two lines control this. There are two gates that both need to pass before the servo moves — that's why it requires shouting, both are currently conservative:

Line 22 — SOUND_ON_THRESH — the cross-correlation gate (primary)
Line 31 — NOISE_MULTIPLIER — the RMS gate (secondary, noise_floor × this)

Start with line 22 only. Flash, test. If still not enough, also touch line 31.

## Line 22 — SOUND_ON_THRESH (change this first)

Level	Value	What it means
Current	500000000000.0f	mean + 3σ — need to shout
Level 1	400000000000.0f	mean + 2.2σ — loud conversation
Level 2	300000000000.0f	mean + 1.2σ — normal conversation
Level 3	220000000000.0f	mean + 0.4σ — quiet speech, some false triggers possible
Level 4	150000000000.0f	below quiet mean — will false trigger on room noise


Each time you change line 22, also change line 23 (SOUND_OFF_THRESH) to 65% of whatever you set on line 22 so hysteresis stays proportional:
## Line 22	Line 23

400000000000.0f	260000000000.0f
300000000000.0f	195000000000.0f
220000000000.0f	143000000000.0f

## Line 31 — NOISE_MULTIPLIER (only if line 22 alone isn't enough)

Level	Value	RMS trigger at
Current	2.5f	~45,000
Level 1	2.0f	~36,000
Level 2	1.7f	~30,600
Level 3	1.5f	~27,000 — borderline, quiet room may trigger

Recommended starting point: set line 22 to Level 2 (300000000000.0f) and line 23 to 195000000000.0f, leave line 31 alone. That should cover normal conversational speech at 1–2 metres.


