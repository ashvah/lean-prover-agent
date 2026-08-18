import Mathlib

/-!
# Phase 0 smoke proofs

Small proofs compiled by `lake build` to check the pinned Lean and Mathlib setup.
-/

set_option linter.style.header false

example (p : Prop) (h : p) : p := by
  exact h

example (x : ℝ) : (x + 1)^2 = x^2 + 2*x + 1 := by
  ring
