import Mathlib

/-
Lean Prover Agent generated inspection file.
source JSONL: retry_smoke_qwen_k4_20260819_073055Z.jsonl
model alias: qwen
model: qwen
strategy: retry
total tasks: 5
verified tasks: 5
failed tasks: 0
Failed model-generated proofs are intentionally preserved.
-/

/-
theorem_id: smoke_exact_001
strategy: retry
selected_attempt: 1 of 1 used
generation_budget: 4
status: VERIFIED
benchmark_verified: true
-/

example (p : Prop) (h : p) : p := by exact h

/-
theorem_id: smoke_exact_002
strategy: retry
selected_attempt: 2 of 2 used
generation_budget: 4
status: VERIFIED
benchmark_verified: true
-/

example (α : Type) (x : α) : x = x := by simp

/-
theorem_id: smoke_intro_003
strategy: retry
selected_attempt: 1 of 1 used
generation_budget: 4
status: VERIFIED
benchmark_verified: true
-/

example (p : Prop) : p → p := by intro h; exact h

/-
theorem_id: smoke_intro_004
strategy: retry
selected_attempt: 1 of 1 used
generation_budget: 4
status: VERIFIED
benchmark_verified: true
-/

example (p q : Prop) : p → q → p := by intro hp hq; exact hp

/-
theorem_id: smoke_constructor_005
strategy: retry
selected_attempt: 1 of 1 used
generation_budget: 4
status: VERIFIED
benchmark_verified: true
-/

example (p q : Prop) (hp : p) (hq : q) : p ∧ q := by exact ⟨hp, hq⟩
