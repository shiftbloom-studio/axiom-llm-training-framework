# Synthetic ML/Software Benchmark Claims

The AlphaBench report is a synthetic fixture for Axiom corpus tests.
AlphaBench uses a fixed temporal split to compare code repair methods.
The baseline model improves compile success on AlphaBench when examples include failing tests.
The same method reduces patch diversity on small repositories.
The dataset card reports that AlphaBench is limited to Python packages with permissive licenses.

The BetaEval note is a synthetic replication note.
BetaEval measures model behavior on issue-triage tasks before release date cutoffs.
BetaEval suggests that repository metadata improves retrieval for issue reproduction.
The replication note contradicts broad claims that metadata always improves repair quality.
Later work supersedes the first BetaEval split with a stricter temporal holdout.

The GammaSuite README describes a software engineering benchmark.
GammaSuite uses containerized tests to measure functional repair outcomes.
GammaSuite shows higher variance when benchmark tasks share the same template.
The benchmark card indicates that task families should be grouped before splitting.
GammaSuite has license metadata for each included repository.
