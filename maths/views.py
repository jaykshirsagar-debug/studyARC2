# maths/views.py
from django.shortcuts import render
from sympy import (
    Eq, Symbol, exp, Interval, S, solveset, sympify,
    pi, simplify, nsimplify
)
from sympy.parsing.latex import parse_latex
from sympy.sets import ConditionSet, Union, ImageSet, FiniteSet
from sympy.core.relational import Relational
from sympy.solvers.inequalities import solve_univariate_inequality
import numpy as np


# ----------------------------------------------------------------------------
# 1. Simple general-solution formatter (plain text)
# ----------------------------------------------------------------------------
def simple_format_general_solution(solset, var_name="x") -> str:
    """
    Plain-text general-solution formatter.
    Tries to give nice forms for common trig infinite sets (ImageSet/Union).
    Falls back to str(solset) if it can't understand.
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
        dummy_syms = list(img.lamda.variables)
        dummy = dummy_syms[0] if dummy_syms else None

        if dummy is None:
            offsets.append(offset_mod_2pi(lam))
            continue

        seen = set()
        sample_offsets = []
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

    def sort_key(expr):
        try:
            return (0, float(expr.evalf()))
        except Exception:
            return (1, str(expr))

    unique_sorted = sorted(unique, key=sort_key)

    # ---- ONE OFFSET CASE ----
    if len(unique_sorted) == 1:
        off = simplify(unique_sorted[0])

        # If the offset is exactly 0, most common cases are:
        #   cos(x) = 1  →  x = 2*pi*n
        #   tan(x) = 0  →  this actually gives two offsets {0, pi}, handled below
        #
        # Our sampling for tan(x)=0 produces {0, pi}, so that goes through
        # the "two offsets" branch, not here. So here, off = 0 is safely the
        # 2*pi*n family.
        if off == 0:
            return f"{var_name} = 2*pi*n, n ∈ ℤ"

        # Generic single offset → offset + 2*pi*n
        return f"{var_name} = {off} + 2*pi*n, n ∈ ℤ"

    # ---- TWO OFFSET CASE ---- (e.g. sin, cos general solutions)
    if len(unique_sorted) == 2:
        a, b = unique_sorted
        try:
            if simplify((b - a) % (2 * pi)) == pi:
                # Families differ by pi → n*pi form
                return f"{var_name} = {a} + n*pi, n ∈ ℤ"
        except Exception:
            pass
        return f"{var_name} = {a} + 2*pi*n or {var_name} = {b} + 2*pi*n, n ∈ ℤ"

    # Fallback
    return str(solset)



# ----------------------------------------------------------------------------
# 2. Numeric root finding (for equations)
# ----------------------------------------------------------------------------
def f_numeric_from_sympy(f, var):
    def f_num(x):
        try:
            return float(f.subs(var, x).evalf())
        except Exception:
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
            continue

        if f1 * f2 < 0:
            brackets.append((x1, x2))

    return brackets


def bisect_root(f_numeric, left, right, tol=1e-10, max_iter=80):
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

        if fl * fm < 0:
            right = mid
            fr = fm
        else:
            left = mid
            fl = fm

    return (left + right) / 2.0


# ----------------------------------------------------------------------------
# 3. Main view — equations + inequalities
# ----------------------------------------------------------------------------
def maths(request):
    context = {}

    if request.method == "POST":
        latex_expr = request.POST.get("expression", "").strip()
        solution_type = request.POST.get("solution_type", "real")
        format_mode = request.POST.get("format_mode", "decimal")

        # RAW domain strings from form
        domain_min_raw = (request.POST.get("domain_min", "-oo") or "").strip()
        domain_max_raw = (request.POST.get("domain_max", "oo") or "").strip()

        context["input_latex"] = latex_expr
        context["solution_type"] = solution_type
        context["format_mode"] = format_mode
        context["domain_min"] = domain_min_raw
        context["domain_max"] = domain_max_raw

        # recognise "infinite" tokens
        inf_tokens_min = {"-oo", "-inf", "-infinity"}
        inf_tokens_max = {"oo", "+oo", "inf", "infinity"}

        # Any infinite bound? (semi-infinite OR full infinite)
        has_infinite_bound = (
            domain_min_raw in inf_tokens_min or
            domain_max_raw in inf_tokens_max
        )

        if latex_expr:
            try:
                # Parse LaTeX from MathQuill
                expr = parse_latex(latex_expr)

                # Treat 'e' and 'pi' as constants
                expr = expr.subs(Symbol("e"), exp(1))
                expr = expr.subs(Symbol("pi"), pi)

                # -------------------------------
                # Parse domain using SymPy safely
                # Map -oo / oo to large finite multiples of pi
                # -------------------------------
                try:
                    if domain_min_raw in inf_tokens_min:
                        a_sym = -40 * pi
                    else:
                        a_sym = sympify(domain_min_raw)

                    if domain_max_raw in inf_tokens_max:
                        b_sym = 40 * pi
                    else:
                        b_sym = sympify(domain_max_raw)
                except Exception:
                    a_sym, b_sym = -40 * pi, 40 * pi

                # Numeric bounds
                a = float(a_sym.evalf())
                b = float(b_sym.evalf())
                if b <= a:
                    b = a + 1.0

                # -------------------------------
                # Variable detection (shared)
                # -------------------------------
                symbols = sorted(expr.free_symbols, key=lambda s: s.name)
                var = None
                if symbols:
                    xs = [s for s in symbols if s.name == "x"]
                    if xs:
                        var = xs[0]
                    else:
                        non_pi = [s for s in symbols if s.name not in ("pi", "π")]
                        var = non_pi[0] if non_pi else symbols[0]
                elif "x" in latex_expr:
                    var = Symbol("x")

                if var is None:
                    context["error"] = "No variable found to solve for."
                    return render(request, "maths/maths.html", context)

                var_name = str(var)

                # -------------------------------
                # CASE 1: INEQUALITY
                # -------------------------------
                if isinstance(expr, Relational) and not isinstance(expr, Eq):
                    sol = solve_univariate_inequality(
                        expr, var, domain=S.Reals, relational=False
                    )

                    dom_interval = Interval(a_sym, b_sym)
                    try:
                        sol_in_dom = sol.intersect(dom_interval)
                    except Exception:
                        sol_in_dom = sol

                    context["equation"] = str(expr)
                    context["inequality_solution"] = str(sol_in_dom)
                    context["variable"] = var_name
                    return render(request, "maths/maths.html", context)

                # -------------------------------
                # CASE 2: EQUATION (or plain expr)
                # -------------------------------
                if isinstance(expr, Eq):
                    equation = expr
                else:
                    equation = Eq(expr, 0)

                context["equation"] = str(equation)
                context["variable"] = var_name

                f = equation.lhs - equation.rhs

                # Global solution set over reals
                solset = solveset(f, var, domain=S.Reals)

                is_finite = isinstance(solset, FiniteSet)
                is_condition = isinstance(solset, ConditionSet)

                # 2A: Infinite/parametric set (ImageSet / Union(...))
                if (not is_finite) and (not is_condition):
                    if has_infinite_bound:
                        # Any infinite bound in the domain → show general solution only
                        context["general_solution"] = simple_format_general_solution(solset, var_name)
                        return render(request, "maths/maths.html", context)
                    else:
                        # Purely finite domain → try to get exact roots inside [a_sym, b_sym]
                        dom_interval = Interval(a_sym, b_sym)
                        interval_solset = solveset(f, var, domain=dom_interval)

                        solutions_display = []

                        if isinstance(interval_solset, FiniteSet):
                            roots_sorted = sorted(interval_solset, key=lambda r: r.evalf())
                            for r in roots_sorted:
                                solutions_display.append(f"{var_name} = {r}")

                            if solutions_display:
                                context["solutions"] = solutions_display
                            else:
                                context["error"] = "No solutions found in the given domain."
                            return render(request, "maths/maths.html", context)
                        else:
                            # Fallback: numeric roots in [a, b]
                            f_numeric = f_numeric_from_sympy(f, var)
                            brackets = bracket_roots(f_numeric, a, b, N=400)

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

                            for r in cleaned:
                                exact = nsimplify(r)
                                solutions_display.append(f"{var_name} = {exact}")

                            if solutions_display:
                                context["solutions"] = solutions_display
                            else:
                                context["error"] = "No solutions found in the given domain."
                            return render(request, "maths/maths.html", context)

                # 2B: ConditionSet → numeric-only in [a, b]
                if is_condition:
                    f_numeric = f_numeric_from_sympy(f, var)
                    brackets = bracket_roots(f_numeric, a, b, N=400)

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

                    solutions_display = []
                    for r in cleaned:
                        if format_mode == "decimal":
                            solutions_display.append(f"{var_name} = {round(r, 10)}")
                        else:
                            exact = nsimplify(r)
                            solutions_display.append(f"{var_name} = {exact}")

                    if solutions_display:
                        context["solutions"] = solutions_display
                    else:
                        context["error"] = "No numeric solutions found in the given domain."
                    return render(request, "maths/maths.html", context)

                # 2C: FiniteSet → filter by [a, b], exact/decimal output
                solutions_display = []
                roots_in_interval = []
                for r in solset:
                    rv = r.evalf()
                    if rv.is_real:
                        val = float(rv)
                        if a - 1e-9 <= val <= b + 1e-9:
                            roots_in_interval.append(val)

                roots_in_interval = sorted(roots_in_interval)
                cleaned = []
                for r in roots_in_interval:
                    if not cleaned or abs(r - cleaned[-1]) > 1e-6:
                        cleaned.append(r)

                for r in cleaned:
                    if format_mode == "decimal":
                        solutions_display.append(f"{var_name} = {round(r, 10)}")
                    else:
                        exact = nsimplify(r)
                        solutions_display.append(f"{var_name} = {exact}")

                if solutions_display:
                    context["solutions"] = solutions_display
                else:
                    context["error"] = "No solutions found in the given domain."

            except Exception as e:
                context["error"] = f"Could not understand the expression: {e}"

        else:
            context["error"] = "Please enter an equation."

    return render(request, "maths/maths.html", context)
