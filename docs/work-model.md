# A work-amplification model for HNSW (threshold model)

The walk/visitation surrogate is the wrong abstraction for work: its gradient
is dead (peaked softmax) and it models an ef=1 trajectory. Work in HNSW is set
by the STOPPING RULE, so model that directly.

## The model

Layer-0 work = distance evaluations ~ sum over expanded nodes of their degree.
A node is expanded iff, when popped, its score beats the current ef-th best, and
iff it is reachable from the entry through other expanded nodes. So work
factorises into an admission term (score vs a threshold) and a reachability
term (graph):

Let R be the reachable candidate region (clean visited set and its neighbours),
s_v(δ) = s(q, e_v(δ)) the score, deg(v) the layer-0 degree.

    θ(δ)   = soft ef-th largest of { s_v(δ) : v ∈ R }        # admission threshold
    a_v(δ) = σ( (s_v(δ) − θ(δ)) / T )                        # soft "v is expanded"
    r_v(δ) = P(v reachable through expanded nodes from entry) # graph gate
    ------------------------------------------------------------------
    W(δ)   = Σ_{v ∈ R} a_v(δ) · r_v(δ) · deg(v)              # expected work
    Amp(δ) = W(δ) / W(0) − 1

θ is a differentiable order statistic, a_v a sigmoid, so W is differentiable in
the scores, and dJ/ds carries the renormalisation exactly as GeoObjective does.

## Why this is the right object

The threshold θ is the whole mechanism, and the model makes every measured
result fall out:

- Tail demotion saturates. Lowering the scores of borderline-admitted nodes
  pushes θ down, but scores are dense near θ, so θ barely moves and Amp caps.
  (Matches 5.8% -> 5.5%.)
- Lures work but compete. Raising an unadmitted node above θ adds a_v·r_v·deg(v)
  directly, but it is now in the top-ef so it RAISES θ, un-admitting the old
  ef-th. Net gain = deg(new) − deg(displaced). This is why lures fight each
  other and why the useful ones sit just above θ (cheap displacement).
- Global small budget fails. One fixed-direction flip cannot raise θ, or admit
  a node, for a distribution of query directions at once. (Matches ~0.4%.)
- Reachability is why "path" (already-expanded) nodes have low leverage: their
  a_v·r_v is already ~1, so ∂W/∂s is ~0 there. Budget belongs on the a_v ∈
  (0,1) boundary, not the saturated head.

## Gradient (the coupling that matters)

    ∂W/∂s_v = a_v' r_v deg(v)                         # direct: admit v more
              + (∂θ/∂s_v) Σ_u (−a_u') r_u deg(u)      # global: v enters top-ef,
                                                       # lifts θ, un-admits others

The second term is the ef-slot competition, absent from the walk model, and it
is what makes the objective honest about displacement.

## Estimating reachability

r_v = 1 for clean-visited v. For a one-hop neighbour of the visited set,
r_v ≈ max over admitted visited neighbours of their admission (a node is
reached if an admitted neighbour points to it). Demoting a node's only admitted
predecessor drops r_v to 0, which is the one way score changes touch the graph
term. Higher-order reachability needs the trace, not a closed form.

## Limits

Exact search only (ANN traversal also changes with codes). r_v beyond one hop
is approximate. deg(v) at layer 0 is near-constant (~2M), so the degree weight
mostly cancels and Amp is driven by the COUNT of newly admitted-and-reachable
nodes, which is the number to report.
