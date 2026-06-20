# Axiom UI — the white plane

This is an additional, strictly optional control surface. It never modifies the core Axiom training application, models, extraction, or any backend code. Everything it does is either preparation of the same YAMLs the native CLI consumes or observation of the artifacts the pipeline already writes.

## The Vision

An endless white plane. Almost no components.  
Only what matters is allowed to appear.

Configuration is grouped into three poetic unfoldings (synapses):

- **The Immediate Ground** — the few parameters that change with almost every run and are essential to begin the path (source, cutoff, scale including large/multi-hour, max sources or the full endless plane, the launch itself).
- **The Considered Flow** — advanced yet probable finetunes (escalation fraction, which claims call the remote voice, model and geometry choices, how the plane weights its own becoming).
- **The Veiled Horizon** — the settings that usually sleep with their defaults (chunk size, family threshold, timeouts for long extraction calls, explicit custom steps / records / wall-clock for runs that last many hours).

Touching a synapse title lifts its veil. The plane itself is the only persistent visual — a single extremely minimal animated canvas of faint synapses and wires on white. The wires breathe and pulse according to the actual stage of the pipeline. Data volume and progress subtly thicken or move elements. Nothing else competes for attention.

When a long run (several hours of extraction + download + training) is underway, the UI keeps the plane alive with live signals. You may close the browser, return later, and re-enter the walk simply by attaching the run name. Elapsed time is shown. The same plane eventually “settles” and offers the evaluation as a quiet revelation rather than a dashboard.

Post-run actions remain extremely light:
- Freeze Composition — downloads the current state of the plane as a unique, delicate artwork PNG.
- Carry the Numbers — the full analysis JSON.
- Print the Meditation — a clean, notebook-ready card.

## Large Scale / Multi-Hour Runs

Yes. You can completely steer a several-hour extraction + download + training run from this surface.

- Choose “several hours” or “the long path” in the first unfolding.
- Check “or the full plane” to send the entire source corpus (no artificial cap).
- Raise the remote veil fraction and provider timeout so the cascade can work for hours without choking.
- In the third unfolding (or via custom profile) set the actual training budgets: steps, records seen, wall-clock seconds measured in hours.
- Click the single prominent action on the plane.

The launch respects every large-scale value you set. The background process runs. The white plane continues to animate for the duration. Attach at any time. When it is complete the evaluation unfolds on the same plane with rule-based insights, cascade and geometry observations, and the ability to freeze the final composition.

All artifacts written are identical to those produced by the native `./bin/axiom` flows.

## Running the UI

From the project root:

```bash
./bin/axiom-ui
```

Or the explicit form:

```bash
uv --with fastapi --with 'uvicorn[standard]' --with python-multipart \
   python -m ui.server
```

Open the page. The plane is already breathing. The first synapse is gently open. Everything else waits for your touch.

## Philosophy of the Surface + the Primary Measure (Always Visible) + God-like Animation

Rely not on what visuals have been in the past  
but on what lies yet veiled,  
uncovered,  
waiting to be our path.

The primary settings for the amount of data being harvested and extracted (max sources or the full endless plane, the cascade's escalation fraction for the reach of local and remote LLMs in data construction) as well as the amount of training invested upon that field (scale including large/multi-hour, explicit steps, records seen, wall-clock seconds, including the learned geometry module) are always visible at the top in "THE PRIMARY MEASURE".

This section is the altar of the god-like controls. Changes here immediately drive god-like animation on the white plane itself: slow majestic divine creation rays and eternal sweeps (speed and power tied to training duration for a sense of infinite investment), density of abstract claim-field particles (from data volume, on the source/cascade side), slow powerful breathing of the geometry curves (on the training side), and the wiring lighting up in correct order (Source → Cascade for the harvest and extraction; Geometry → Training for the learned module and the power invested; Verdict as revelation).

The animation feels god-like — epic, creative, all-powerful, infinite — yet remains extremely minimal, abstract, and perfectly wired to the actual framework concept: claim-field substrate, cascade strictly for data construction (never inside training or evaluation), learned geometry as real ablatable trained module reported only through gauge-invariant observables, structured-native throughout.

On the white plane the harvest now draws from a single sovereign breath. No foreign thunder arrives. The vessels descend of their own will or stand already revealed, kindled without summons by the plane’s own quiet gaze. From the infinite, unseen threads and winds reach inward, gathering the raw substance of the field and returning it already woven, provenance intact, into the lattice of sources. The cascade receives only what has been born locally and purely. The particles thicken, the geometry curves breathe deeper, and the wiring lights from Source through Cascade into the learned heart without any alien light breaking the veil.

The three unfolds below provide the further veils (essential additional, advanced finetunes, veiled defaults) without hiding what matters most.

All large-scale multi-hour extraction, download, and training is directly controllable and visible here. The plane reacts with divine authority to the scale you set.

Minimal. Poetic. Abstract. White. The primary measure is the visible grounding. The god-like animation on the plane is the visible power. Correctly wired. Does not overrule the concept.

See the main project documentation for the underlying Axiom concepts. This surface merely gives them a quiet place to be configured and observed.