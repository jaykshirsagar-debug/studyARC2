# maths/views.py
from django.shortcuts import render

from sympy import (
    Eq, Symbol, exp, Interval, S, solveset, sympify,
    pi, simplify, nsimplify, Function, diff, integrate
)
from sympy.parsing.latex import parse_latex
from sympy.sets import ConditionSet, Union, ImageSet, FiniteSet
from sympy.core.relational import Relational
from sympy.solvers.inequalities import solve_univariate_inequality
import numpy as np
import re


# ----------------------------------------------------------------------------
# 0. def f(x)=... parser
# ----------------------------------------------------------------------------
import re

def extract_def_command(raw: str):
    s = raw.strip()

    if not s.lower().startswith("def"):
        return None

    # Split ONCE at the first "=" (keep RHS intact)
    if "=" not in s:
        raise ValueError("Use: def f(x)=...")

    left_raw, rhs_raw = s.split("=", 1)

    # -------------------------
    # Clean LEFT (for fname/arg)
    # -------------------------
    left = left_raw.strip()

    # remove 'def' and LaTeX wrappers
    left = left[3:]  # after 'def'
    left = left.replace(r"\left", "").replace(r"\right", "")
    left = left.replace(r"\ ", " ")
    left = left.replace("\\", "")  # LEFT ONLY: turn '\ g' into ' g'
    left = re.sub(r"\s+", "", left)  # remove spaces

    # Expect: f(x)
    if "(" not in left or ")" not in left:
        raise ValueError("Use: def f(x)=...")

    fname = left.split("(", 1)[0]
    arg = left.split("(", 1)[1].split(")", 1)[0]

    if not fname.isidentifier():
        raise ValueError("Function name must be a valid identifier (e.g. f, g, h).")
    if not arg.isidentifier():
        raise ValueError("Function argument must be a single variable like x.")

    # -------------------------
    # Clean RHS (keep LaTeX!)
    # -------------------------
    rhs = rhs_raw.strip()
    rhs = rhs.replace(r"\left", "").replace(r"\right", "")
    # keep backslashes so \sin stays \sin
    # also normalize LaTeX-space to normal space
    rhs = rhs.replace(r"\ ", " ")

    if not rhs:
        raise ValueError("Function body is empty.")

    return fname, arg, rhs



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

        # offset = 0 → cos(x)=1 type → 2*pi*n
        if off == 0:
            return f"{var_name} = 2*pi*n, n ∈ ℤ"

        return f"{var_name} = {off} + 2*pi*n, n ∈ ℤ"

    # ---- TWO OFFSET CASE ----
    if len(unique_sorted) == 2:
        a, b = unique_sorted
        try:
            if simplify((b - a) % (2 * pi)) == pi:
                return f"{var_name} = {a} + n*pi, n ∈ ℤ"
        except Exception:
            pass
        return f"{var_name} = {a} + 2*pi*n or {var_name} = {b} + 2*pi*n, n ∈ ℤ"

    return str(solset)


# ----------------------------------------------------------------------------
# 2. Numeric root finding (for ConditionSet equations)
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
# 3. Main view — def/functions + evaluation + eq/ineq
# ----------------------------------------------------------------------------
def maths(request):
    context = {}

    if request.method == "POST":
        latex_expr = (request.POST.get("expression", "") or "").strip()
        format_mode = request.POST.get("format_mode", "decimal")
        domain_min = request.POST.get("domain_min", "-oo")
        domain_max = request.POST.get("domain_max", "oo")

        context["input_latex"] = latex_expr
        context["format_mode"] = format_mode
        context["domain_min"] = domain_min
        context["domain_max"] = domain_max

        if not latex_expr:
            context["error"] = "Please enter an expression."
            return render(request, "maths/maths.html", context)

        # -------------------------------
        # 3A) DEF COMMAND (must start with def)
        # -------------------------------
        try:
            def_cmd = extract_def_command(latex_expr)
        except Exception as e:
            context["error"] = str(e)
            return render(request, "maths/maths.html", context)

        if def_cmd:
            fname, arg, rhs = def_cmd

            # parse RHS as LaTeX
            var = Symbol(arg)
            body = parse_latex(rhs)

            # constants
            body = body.subs(Symbol("e"), exp(1))
            body = body.subs(Symbol("pi"), pi)

            funcs = request.session.get("functions", {})
            funcs[fname] = (arg, str(body))  # store as strings for session safety
            request.session["functions"] = funcs

            context["equation"] = latex_expr
            context["direct_result"] = f"Defined: {fname}({arg}) = {str(body)}"
            return render(request, "maths/maths.html", context)

        # -------------------------------
        # 3B) Parse expression normally
        # -------------------------------
        try:
            expr = parse_latex(latex_expr)

            # constants
            expr = expr.subs(Symbol("e"), exp(1))
            expr = expr.subs(Symbol("pi"), pi)

            # Substitute user-defined functions from session
            funcs = request.session.get("functions", {})
            for name, (argname, body_str) in funcs.items():
                arg_sym = Symbol(argname)
                body_sym = sympify(body_str)

                f = Function(name)
                expr = expr.replace(f, lambda a: body_sym.subs(arg_sym, a))

            # -----------------------------------------
            # Domain bounds (allow -oo/oo/pi/etc)
            # -----------------------------------------
            a_sym = sympify(domain_min)
            b_sym = sympify(domain_max)

            # Build Interval for inequalities (if possible)
            try:
                interval_domain = Interval(a_sym, b_sym)
            except Exception:
                interval_domain = None

            # -------------------------------
            # 3C) diff / integrate / antidiff
            # -------------------------------
            if expr.func.__name__ in {"diff", "integrate", "antidiff"}:
                args = expr.args
                vars_found = list(expr.free_symbols)
                var = vars_found[0] if vars_found else Symbol("x")

                if expr.func.__name__ == "diff":
                    if len(args) == 1:
                        result = simplify(diff(args[0], var))
                    else:
                        result = simplify(diff(*args))

                elif expr.func.__name__ == "antidiff":
                    # treat as indefinite integral
                    if len(args) == 1:
                        result = simplify(integrate(args[0], var))
                    else:
                        result = simplify(integrate(*args))

                else:  # integrate(...)
                    result = simplify(integrate(*args))

                if format_mode == "decimal":
                    result = result.evalf()

                context["equation"] = latex_expr
                context["direct_result"] = str(result)
                return render(request, "maths/maths.html", context)

            # -------------------------------
            # 3D) Pure numeric evaluation
            # -------------------------------
            if not expr.free_symbols and not isinstance(expr, Relational):
                val = simplify(expr) if format_mode == "standard" else expr.evalf()
                context["equation"] = latex_expr
                context["direct_result"] = str(val)
                return render(request, "maths/maths.html", context)

            # -------------------------------
            # 3E) Inequalities
            # -------------------------------
            if isinstance(expr, Relational) and not isinstance(expr, Eq):
                symbols = sorted(expr.free_symbols, key=lambda s: s.name)
                var = symbols[0] if symbols else Symbol("x")

                sol = solve_univariate_inequality(expr, var, domain=S.Reals, relational=False)

                # Intersect with interval if the user gave finite bounds
                if interval_domain is not None and interval_domain != Interval(-S.Infinity, S.Infinity):
                    try:
                        sol = sol.intersect(interval_domain)
                    except Exception:
                        pass

                context["equation"] = str(expr)
                context["inequality_solution"] = str(sol)
                return render(request, "maths/maths.html", context)

            # -------------------------------
            # 3F) Equations (or plain expr -> =0)
            # -------------------------------
            if isinstance(expr, Eq):
                equation = expr
            else:
                equation = Eq(expr, 0)

            # variable
            symbols = sorted(equation.free_symbols, key=lambda s: s.name)
            var = symbols[0] if symbols else Symbol("x")

            f = equation.lhs - equation.rhs
            solset = solveset(f, var, domain=S.Reals)

            # Infinite -> general
            if not isinstance(solset, FiniteSet) and not isinstance(solset, ConditionSet):
                context["equation"] = str(equation)
                context["general_solution"] = simple_format_general_solution(solset, str(var))
                return render(request, "maths/maths.html", context)

            # ConditionSet -> numeric scan only if bounds are finite
            if isinstance(solset, ConditionSet):
                # Need finite numeric bounds for scanning
                try:
                    a = float(a_sym.evalf())
                    b = float(b_sym.evalf())
                except Exception:
                    context["equation"] = str(equation)
                    context["general_solution"] = f"Solution set described by: {solset}"
                    return render(request, "maths/maths.html", context)

                if b <= a:
                    b = a + 1.0

                f_numeric = f_numeric_from_sympy(f, var)
                brackets = bracket_roots(f_numeric, a, b, N=500)

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

                if not cleaned:
                    context["error"] = "No numeric solutions found in the given domain."
                    context["equation"] = str(equation)
                    return render(request, "maths/maths.html", context)

                solutions_display = []
                for r in cleaned:
                    if format_mode == "decimal":
                        solutions_display.append(f"{var} = {round(r, 10)}")
                    else:
                        solutions_display.append(f"{var} = {nsimplify(r)}")

                context["equation"] = str(equation)
                context["solutions"] = solutions_display
                return render(request, "maths/maths.html", context)

            # FiniteSet -> list solutions (optionally filter if bounds are finite)
            sols = list(solset)

            # If bounds are finite numeric, filter by them. If -oo/oo, show all.
            finite_filter = True
            try:
                a = float(a_sym.evalf())
                b = float(b_sym.evalf())
            except Exception:
                finite_filter = False

            filtered = []
            if finite_filter:
                lo, hi = (a, b) if a <= b else (b, a)
                for s in sols:
                    sv = s.evalf()
                    if sv.is_real:
                        v = float(sv)
                        if lo - 1e-9 <= v <= hi + 1e-9:
                            filtered.append(s)
            else:
                filtered = sols

            if not filtered:
                # If user gave finite interval and nothing fits, say so
                context["error"] = "No solutions found in the given domain."
                context["equation"] = str(equation)
                return render(request, "maths/maths.html", context)

            solutions_display = []
            for s in filtered:
                if format_mode == "decimal":
                    solutions_display.append(f"{var} = {s.evalf()}")
                else:
                    solutions_display.append(f"{var} = {s}")

            context["equation"] = str(equation)
            context["solutions"] = solutions_display
            return render(request, "maths/maths.html", context)

        except Exception as e:
            context["error"] = f"Could not understand the expression: {e}"
            return render(request, "maths/maths.html", context)

    return render(request, "maths/maths.html", context)
