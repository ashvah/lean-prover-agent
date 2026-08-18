import Mathlib

/-!
# Phase 1 smoke dataset sanitation

Reference proofs compiled by `lake build` verify the statements in `data/smoke.jsonl`
against the pinned Lean and Mathlib environment. The model runner does not receive them.
-/

set_option linter.style.header false

example (p : Prop) (h : p) : p := by
  exact h

example (α : Type) (x : α) : x = x := by
  rfl

example (p : Prop) : p → p := by
  intro hp
  exact hp

example (p q : Prop) : p → q → p := by
  intro hp _
  exact hp

example (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  constructor
  · exact hp
  · exact hq

example (p q : Prop) : p ∧ q → q ∧ p := by
  intro h
  constructor
  · exact h.2
  · exact h.1

example (p : Prop) : p ↔ p := by
  constructor
  · intro hp
    exact hp
  · intro hp
    exact hp

example (α : Type) (x : α) : ∃ y, y = x := by
  exact ⟨x, rfl⟩

example (n : Nat) : n + 0 = n := by
  simp

example (α : Type) (xs : List α) : xs ++ [] = xs := by
  simp

example (p : Prop) : p ∧ True ↔ p := by
  simp

example (x : Int) : x - x = 0 := by
  simp

example (a b : Nat) (h : a = b) : a + 1 = b + 1 := by
  rw [h]

example (p q : Prop) (h : p = q) (hp : p) : q := by
  rw [h] at hp
  exact hp

example : (2 : Nat) + 3 = 5 := by
  norm_num

example : (7 : Nat) * 6 = 42 := by
  norm_num

example : (3 : Int) ^ 2 = 9 := by
  norm_num

example (x : ℝ) : (x + 1) ^ 2 = x ^ 2 + 2 * x + 1 := by
  ring

example (x y : ℝ) : (x + y) * (x - y) = x ^ 2 - y ^ 2 := by
  ring

example (x : ℝ) : (x + 2) * (x + 3) = x ^ 2 + 5 * x + 6 := by
  ring

example (x y : ℝ) (h : x ≤ y) : x + 1 ≤ y + 1 := by
  linarith

example (x y : ℝ) (hxy : x ≤ y) (hyx : y ≤ x) : x = y := by
  linarith

example (x : ℝ) (h : 2 * x = 6) : x = 3 := by
  linarith

example (m n : Nat) (h : m ≤ n) : m ≤ n + 1 := by
  omega

example (p q : Prop) (hp : p) (hq : q) : q ∧ p := by
  aesop
