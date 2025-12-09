# maths/views.py
from django.shortcuts import render
from sympy import (
    Eq, Symbol, exp, Interval, S, solveset, sympify,
    pi, simplify
)
from sympy.parsing.latex import parse_latex
from sympy.sets import ConditionSet, Union, ImageSet, FiniteSet
import numpy as np


# ----------------------------------------------------------------------------
# 1. Simple general-solution formatter (plain text)
# ----------------------------------------------------------------------------
def simple_format_general_solution(solset, var_name="x") -> str:
    """
    Plain-text general-solution formatter.
    Keeps it simple, handles common trig infinite sets where SymPy
    returns ImageSet/Union(ImageSet, ...). Otherwise falls back to str().
    """
    # Finite discrete set -> list them
    if isinstance(solset, FiniteSet):
        vals = list(solset)
        return ", ".join(f"{var_name} = {v}" for v in vals)

    # ConditionSet or unknown structure -> just show the raw SymPy set
    if isinstance(solset, ConditionSet):
        return f"Solution set described by: {solset}"

    # Collect ImageSet objects (for many trig equations)
    def extract_imagesets(s):
        imgs = []
        if isinstance(s, ImageSet):
            imgs.append(s)
        elif isinstance(s, Union):
            for subset in s.args:
                if isinstance(subset, ImageSet):
                    imgs.append(subset)
        return imgs

    imgs = extract_imagesets(solset)
    if not imgs:
        return str(solset)

    # Helper: constant offset modulo 2π
    def offset_mod_2pi(expr):
        expr_s = simplify(expr)
        if expr_s.is_Add:
            for term in expr_s.args:
                if term.free_symbols == set():
                    return simplify(term % (2 * pi))
            return 0
        return simplify(expr_s % (2 * pi))

    offsets = []
    for img in imgs:
        lam = img.lamda.expr
        dummy_syms = list(img.lamda.variables)
        dummy = dummy_syms[0] if dummy_syms else None

        if dummy is None:
            offsets.append(offset_mod_2pi(lam))
            continue

        seen = set()
        sample_offsets = []
        # sample a few integer values of the dummy to discover offsets
        for m in range(0, 8):
            try:
                sampled = lam.subs(dummy, m)
                off = offset_mod_2pi(sampled)
            except Exception:
                continue
            key = str(off)
            if key not in seen:
                seen.add(key)
                sample_offsets.append(off)

        offsets.extend(sample_offsets)

    # Normalize and dedupe offsets into [0, 2π)
    unique = []
    seen = set()
    for o in offsets:
        o_s = simplify(o)
        try:
            norm = simplify(o_s % (2 * pi))
        except Exception:
            norm = o_s
        key = str(norm)
        if key not in seen:
            seen.add(key)
            unique.append(norm)

    if not unique:
        return str(solset)

    # Sort offsets: try numeric eval, else fall back to string
    def sort_key(expr):
        try:
            return (0, float(expr.evalf()))
        except Exception:
            return (1, str(expr))

    unique_sorted = sorted(unique, key=sort_key)

    # One offset
    if len(unique_sorted) == 1:
        off = unique_sorted[0]
        try:
            if simplify(off % pi) == 0:
                return f"{var_name} = n*pi, n ∈ ℤ"
        except Exception:
            pass
        return f"{var_name} = {off} + 2*pi*n, n ∈ ℤ"

    # Two offsets → maybe differ by π
    if len(unique_sorted) == 2:
        a, b = unique_sorted
        try:
            if simplify((b - a) % (2 * pi)) == pi:
                return f"{var_name} = {a} + n*pi, n ∈ ℤ"
        except Exception:
            pass
        return f"{var_name} = {a} + 2*pi*n or {var_name} = {b} + 2*pi*n, n ∈ ℤ"

    # More than two families → fallback
    return str(solset)


# ----------------------------------------------------------------------------
# 2. Numeric root finding for interval mode (SymPy evalf-based)
# ----------------------------------------------------------------------------
def f_numeric_from_sympy(f, var):
    """
    Build a safe numeric evaluator: x (float) -> float(f(x)).
    Uses SymPy subs + evalf and converts to Python float.
    """
    def f_num(x):
        try:
            val = f.subs(var, x).evalf()
            return float(val)
        except Exception:
            return float("nan")
    return f_num


def bracket_roots(f_numeric, a, b, N=400):
    """
    Divide [a, b] into N segments, detect sign changes f(x_i)*f(x_{i+1}) < 0.
    Return a list of (left, right) brackets that each contain at least one root.
    """
    xs = np.linspace(a, b, N + 1)
    fs = [f_numeric(x) for x in xs]

    brackets = []
    for i in range(N):
        x1, x2 = xs[i], xs[i + 1]
        f1, f2 = fs[i], fs[i + 1]

        if np.isnan(f1) or np.isnan(f2):
            continue

        # exact root at grid point
        if abs(f1) < 1e-12:
            brackets.append((x1 - 1e-6, x1 + 1e-6))
            continue

        # sign change
        if f1 * f2 < 0:
            brackets.append((x1, x2))

    return brackets


def bisect_root(f_numeric, left, right, tol=1e-10, max_iter=80):
    """
    Given f_numeric and an interval [left, right] with opposite signs,
    use bisection to find a root. Guaranteed to converge for continuous f.
    """
    fl = f_numeric(left)
    fr = f_numeric(right)

    if np.isnan(fl) or np.isnan(fr) or fl * fr > 0:
        return None

    for _ in range(max_iter):
        mid = (left + right) / 2.0
        fm = f_numeric(mid)

        if np.isnan(fm):
            return None

        if abs(fm) < tol:
            return mid

        # choose the half that contains a sign change
        if fl * fm < 0:
            right = mid
            fr = fm
        else:
            left = mid
            fl = fm

    return (left + right) / 2.0


# ----------------------------------------------------------------------------
# 3. Main view
# ----------------------------------------------------------------------------
def maths(request):
    context = {}

    if request.method == "POST":
        latex_expr = request.POST.get("expression", "").strip()
        solution_type = request.POST.get("solution_type", "real")    # kept for future; interval = real only
        solve_mode = request.POST.get("solve_mode", "interval")      # "interval" or "general"
        domain_min = request.POST.get("domain_min", "-10")
        domain_max = request.POST.get("domain_max", "10")

        context["input_latex"] = latex_expr
        context["solution_type"] = solution_type
        context["solve_mode"] = solve_mode
        context["domain_min"] = domain_min
        context["domain_max"] = domain_max

        if latex_expr:
            try:
                # Parse LaTeX from MathQuill
                expr = parse_latex(latex_expr)

                # Map textual 'e' and 'pi' to their mathematical constants
                expr = expr.subs(Symbol("e"), exp(1))
                expr = expr.subs(Symbol("pi"), pi)  # IMPORTANT: turn a Symbol 'pi' into constant pi

                # Ensure we have an equation; if user typed just an expression, solve expr = 0
                if isinstance(expr, Eq):
                    equation = expr
                else:
                    equation = Eq(expr, 0)

                # Robust variable selection:
                #  1. Prefer a symbol literally named 'x' if present.
                #  2. Otherwise, pick first symbol that is not named 'pi' or 'π'.
                symbols = sorted(equation.free_symbols, key=lambda s: s.name)
                var = None
                if symbols:
                    # prefer x
                    xs = [s for s in symbols if s.name == "x"]
                    if xs:
                        var = xs[0]
                    else:
                        # skip any pi-like symbols if they still exist
                        non_pi = [s for s in symbols if s.name not in ("pi", "π")]
                        var = non_pi[0] if non_pi else symbols[0]
                elif "x" in latex_expr:
                    var = Symbol("x")

                if var is None:
                    context["error"] = "No variable found to solve for."
                    return render(request, "maths/maths.html", context)

                var_name = str(var)
                context["variable"] = var_name
                context["equation"] = str(equation)

                # Work with f(x) = LHS - RHS
                f = equation.lhs - equation.rhs

                # =============================
                # MODE: GENERAL SOLUTION
                # =============================
                if solve_mode == "general":
                    solset = solveset(f, var, domain=S.Reals)
                    context["general_solution"] = simple_format_general_solution(solset, var_name)
                    return render(request, "maths/maths.html", context)

                # =============================
                # MODE: INTERVAL SOLUTION
                # =============================
                # Parse domain bounds safely
                try:
                    a = float(sympify(domain_min).evalf())
                    b = float(sympify(domain_max).evalf())
                except Exception:
                    a, b = -10.0, 10.0

                if b <= a:
                    b = a + 1.0  # avoid zero-length interval

                # Build numeric evaluator using SymPy subs + evalf
                f_numeric = f_numeric_from_sympy(f, var)

                # 1) Find brackets with sign changes
                brackets = bracket_roots(f_numeric, a, b, N=400)

                # 2) Bisection in each bracket
                roots = []
                for left, right in brackets:
                    r = bisect_root(f_numeric, left, right)
                    if r is not None:
                        roots.append(r)

                # 3) Deduplicate roots (neighbouring brackets may converge to the same root)
                roots_sorted = sorted(roots)
                cleaned = []
                for r in roots_sorted:
                    if not cleaned or abs(r - cleaned[-1]) > 1e-6:
                        cleaned.append(r)

                if cleaned:
                    context["solutions"] = [f"{var_name} = {round(r, 10)}" for r in cleaned]
                else:
                    context["error"] = "No solutions found in the given domain."

            except Exception as e:
                context["error"] = f"Could not understand the expression: {e}"

        else:
            context["error"] = "Please enter an equation."

    return render(request, "maths/maths.html", context)
