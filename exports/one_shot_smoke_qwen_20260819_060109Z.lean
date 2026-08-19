import Mathlib

/-
Lean Prover Agent generated inspection file.
source JSONL: one_shot_smoke_qwen_20260819_060109Z.jsonl
model alias: qwen
model: qwen
strategy: one_shot
total tasks: 5
verified tasks: 5
failed tasks: 0
Failed model-generated proofs are intentionally preserved.
-/

/-
theorem_id: smoke_exact_001
status: VERIFIED
benchmark_verified: true
-/

example (p : Prop) (h : p) : p := by exact h

/-
theorem_id: smoke_exact_002
status: VERIFIED
benchmark_verified: true
-/

example (α : Type) (x : α) : x = x := by rfl

/-
theorem_id: smoke_intro_003
status: VERIFIED
benchmark_verified: true
-/

example (p : Prop) : p → p := by intro hp; exact hp

/-
theorem_id: smoke_intro_004
status: VERIFIED
benchmark_verified: true
-/

example (p q : Prop) : p → q → p := by intro h1 h2; exact h1

/-
theorem_id: smoke_constructor_005
status: VERIFIED
benchmark_verified: true
-/

example (p q : Prop) (hp : p) (hq : q) : p ∧ q := by exact ⟨hp, hq⟩
