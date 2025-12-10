# maths/views.py
from django.shortcuts import render
from sympy import (
    Eq, Symbol, exp, Interval, S, solveset, sympify,
    pi, simplify, nsimplify
)
from sympy.parsing.latex import parse_latex
from sympy.sets import ConditionSet, Union, ImageSet, FiniteSet
import numpy as np


# ----------------------------------------------------------------------------
# 1. Simple general-solution formatter (plain text)
# ----------------------------------------------------------------------------
def simple_format_general_solution(solset, var_name="x") -> str:
    """
    Same as your original formatter, unchanged.
    """
    if isinstance(solset, FiniteSet):
        vals = list(solset)
        return ", ".join(f"{var_name} = {v}" for v in vals)

    if isinstance(solset, ConditionSet):
        return f"Solution set described by: {solset}"

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
        dummy = list(img.lamda.variables)[0]

        seen = set()
        for m in range(8):
            try:
                sampled = lam.subs(dummy, m)
                off = offset_mod_2pi(sampled)
            except:
                continue
            if str(off) not in seen:
                seen.add(str(off))
                offsets.append(off)

    # Normalize & dedupe
    uniq = []
    seen = set()
    for o in offsets:
        try:
            norm = simplify(o % (2*pi))
        except:
            norm = simplify(o)
        if str(norm) not in seen:
            seen.add(str(norm))
            uniq.append(norm)

    if not uniq:
        return str(solset)

    # Sort
    def srt(expr):
        try:
            return float(expr.evalf())
        except:
            return str(expr)

    uniq = sorted(uniq, key=srt)

    if len(uniq) == 1:
        off = uniq[0]
        return f"{var_name} = {off} + 2*pi*n, n ∈ ℤ"

    if len(uniq) == 2:
        a, b = uniq
        return f"{var_name} = {a} + 2*pi*n or {var_name} = {b} + 2*pi*n, n ∈ ℤ"

    return str(solset)


# ----------------------------------------------------------------------------
# 2. Numeric root finding (interval mode)
# ----------------------------------------------------------------------------
def f_numeric_from_sympy(f, var):
    def f_num(x):
        try:
            return float(f.subs(var, x).evalf())
        except:
            return float("nan")
    return f_num


def bracket_roots(f_numeric, a, b, N=400):
    xs = np.linspace(a, b, N + 1)
    fs = [f_numeric(x) for x in xs]

    brackets = []
    for i in range(N):
        x1, x2 = xs[i], xs[i + 1]
        f1, f2 = fs[i], fs[i + 1]

        if np.isnan(f1) or np.isnan(f2):
            continue

        if abs(f1) < 1e-12:
            brackets.append((x1 - 1e-6, x1 + 1e-6))
        elif f1 * f2 < 0:
            brackets.append((x1, x2))

    return brackets


def bisect_root(f_numeric, left, right, tol=1e-10, max_iter=80):
    fl = f_numeric(left)
    fr = f_numeric(right)
    if np.isnan(fl) or np.isnan(fr) or fl * fr > 0:
        return None

    for _ in range(max_iter):
        mid = (left + right) / 2
        fm = f_numeric(mid)

        if np.isnan(fm):
            return None
        if abs(fm) < tol:
            return mid

        if fl * fm < 0:
            right = mid
            fr = fm
        else:
            left = mid
            fl = fm

    return (left + right) / 2


# ----------------------------------------------------------------------------
# 3. Main view — interval + general mode, now with format_mode
# ----------------------------------------------------------------------------
def maths(request):
    context = {}

    if request.method == "POST":
        latex_expr = request.POST.get("expression", "").strip()
        solution_type = request.POST.get("solution_type", "real")
        solve_mode = request.POST.get("solve_mode", "interval")
        format_mode = request.POST.get("format_mode", "decimal")   # <<<< NEW
        domain_min = request.POST.get("domain_min", "-10")
        domain_max = request.POST.get("domain_max", "10")

        context["input_latex"] = latex_expr
        context["solution_type"] = solution_type
        context["solve_mode"] = solve_mode
        context["format_mode"] = format_mode
        context["domain_min"] = domain_min
        context["domain_max"] = domain_max

        if latex_expr:
            try:
                expr = parse_latex(latex_expr)
                expr = expr.subs(Symbol("e"), exp(1))
                expr = expr.subs(Symbol("pi"), pi)

                if isinstance(expr, Eq):
                    equation = expr
                else:
                    equation = Eq(expr, 0)

                symbols = sorted(equation.free_symbols, key=lambda s: s.name)

                xs = [s for s in symbols if s.name == "x"]
                if xs:
                    var = xs[0]
                else:
                    non_pi = [s for s in symbols if s.name not in ("pi", "π")]
                    var = non_pi[0] if non_pi else Symbol("x")

                var_name = str(var)
                context["variable"] = var_name
                context["equation"] = str(equation)

                f = equation.lhs - equation.rhs

                # GENERAL MODE
                if solve_mode == "general":
                    solset = solveset(f, var, domain=S.Reals)
                    context["general_solution"] = simple_format_general_solution(solset, var_name)
                    return render(request, "maths/maths.html", context)

                # INTERVAL MODE
                a = float(sympify(domain_min).evalf())
                b = float(sympify(domain_max).evalf())
                if b <= a:
                    b = a + 1

                f_numeric = f_numeric_from_sympy(f, var)
                brackets = bracket_roots(f_numeric, a, b)

                roots = []
                for left, right in brackets:
                    r = bisect_root(f_numeric, left, right)
                    if r is not None:
                        roots.append(r)

                roots_sorted = sorted(roots)
                cleaned = []
                for r in roots_sorted:
                    if not cleaned or abs(r - cleaned[-1]) > 1e-6:
                        cleaned.append(r)

                # NOW APPLY DECIMAL vs STANDARD MODE
                display = []
                for r in cleaned:
                    if format_mode == "decimal":
                        display.append(f"{var_name} = {round(r, 10)}")
                    else:
                        exact = nsimplify(r)   # <<<< AGGRESSIVE S2 MODE
                        display.append(f"{var_name} = {exact}")

                if display:
                    context["solutions"] = display
                else:
                    context["error"] = "No solutions found in the given domain."

            except Exception as e:
                context["error"] = f"Could not understand the expression: {e}"

        else:
            context["error"] = "Please enter an equation."

    return render(request, "maths/maths.html", context)
