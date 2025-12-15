# maths/views.py
from django.shortcuts import render
from sympy import (
    Eq, Symbol, exp, S, solveset, sympify,
    pi, simplify, nsimplify, Function, diff, integrate
)
from sympy.parsing.latex import parse_latex
from sympy.sets import FiniteSet, ConditionSet
from sympy.core.relational import Relational
from sympy.solvers.inequalities import solve_univariate_inequality
import re


# -----------------------------
# Numeric helpers (bisection)
# -----------------------------
def f_numeric_from_sympy(expr, var):
    def f(x):
        return float(expr.subs(var, x).evalf())
    return f


def bracket_roots(f, a, b, N=800):
    brackets = []
    step = (b - a) / N
    x0 = a
    try:
        f0 = f(x0)
    except Exception:
        f0 = None

    for i in range(1, N + 1):
        x1 = a + i * step
        try:
            f1 = f(x1)
        except Exception:
            f1 = None

        if f0 is not None and f1 is not None:
            # root exactly at grid point
            if abs(f0) < 1e-10:
                brackets.append((x0 - 1e-6, x0 + 1e-6))
            # sign change
            elif f0 * f1 < 0:
                brackets.append((x0, x1))

        x0, f0 = x1, f1

    return brackets


def bisect_root(f, a, b, tol=1e-10, max_iter=100):
    try:
        fa = f(a)
        fb = f(b)
    except Exception:
        return None

    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        return None

    left, right = a, b
    for _ in range(max_iter):
        mid = (left + right) / 2.0
        try:
            fm = f(mid)
        except Exception:
            return None

        if abs(fm) < tol or abs(right - left) < tol:
            return mid

        if fa * fm < 0:
            right = mid
            fb = fm
        else:
            left = mid
            fa = fm

    return (left + right) / 2.0


def clean_roots(roots, eps=1e-6):
    roots = sorted(roots)
    out = []
    for r in roots:
        if not out or abs(r - out[-1]) > eps:
            out.append(r)
    return out


# -----------------------------
# Casio-style adaptive window
# -----------------------------
def adaptive_numeric_solve(f_expr, var, domain_min_sym, domain_max_sym):
    """
    If domain is infinite (-oo..oo), scan in expanding windows until:
      - no new roots appear AND
      - no roots are near the boundary
    """
    # If user gave finite bounds, just use them
    is_min_inf = (domain_min_sym == -S.Infinity)
    is_max_inf = (domain_max_sym == S.Infinity)

    if not (is_min_inf and is_max_inf):
        # finite or semi-infinite (we only handle finite numerically here)
        try:
            a = float(domain_min_sym.evalf())
            b = float(domain_max_sym.evalf())
            if b <= a:
                b = a + 1.0
            return (a, b, False)
        except Exception:
            # can't numerically scan semi-infinite cleanly
            return (None, None, False)

    # Infinite case: start like Casio
    start = 10.0
    max_abs = 640.0  # safety cap
    max_rounds = 8
    edge_ratio = 0.06  # "near boundary" if within 6% of window edge

    f_num = f_numeric_from_sympy(f_expr, var)

    prev = []
    L = start

    for _ in range(max_rounds):
        a, b = -L, L

        brackets = bracket_roots(f_num, a, b, N=1200)
        roots = []
        for left, right in brackets:
            r = bisect_root(f_num, left, right)
            if r is not None:
                roots.append(r)

        roots = clean_roots(roots, eps=1e-6)
        # Normalize for stability comparison (rounding)
        roots_key = tuple(round(r, 8) for r in roots)

        # Check if roots are hugging boundaries → expand
        near_edge = False
        if roots:
            margin = edge_ratio * (b - a)  # e.g. ~1.2 when L=10
            for r in roots:
                if abs(r - a) < margin or abs(b - r) < margin:
                    near_edge = True
                    break

        # Stable if no new roots AND not near edges
        prev_key = tuple(round(r, 8) for r in prev)
        if roots_key == prev_key and not near_edge:
            return (a, b, True)

        prev = roots
        L *= 2.0
        if L > max_abs:
            break

    # give back the last attempted window
    return (-min(L, max_abs), min(L, max_abs), True)


# -----------------------------
# def f(x)=... parser
# -----------------------------
def extract_def_command(raw):
    s = raw.strip()
    if not s.lower().startswith("def"):
        return None

    if "=" not in s:
        raise ValueError("Use: def f(x)=...")

    left_raw, rhs_raw = s.split("=", 1)

    # Clean LEFT: keep it identifier-safe
    left = left_raw.strip()[3:]  # after 'def'
    left = left.replace(r"\left", "").replace(r"\right", "")
    left = left.replace(r"\ ", " ")
    left = left.replace("\\", "")  # only on left
    left = re.sub(r"\s+", "", left)

    if "(" not in left or ")" not in left:
        raise ValueError("Use: def f(x)=...")

    fname = left.split("(", 1)[0]
    arg = left.split("(", 1)[1].split(")", 1)[0]

    if not fname.isidentifier():
        raise ValueError("Function name must be a valid identifier (e.g. f, g, h).")
    if not arg.isidentifier():
        raise ValueError("Function argument must be a single variable like x.")

    rhs = rhs_raw.strip()
    rhs = rhs.replace(r"\left", "").replace(r"\right", "")
    rhs = rhs.replace(r"\ ", " ")

    if not rhs:
        raise ValueError("Function body is empty.")

    return fname, arg, rhs


# -----------------------------
# General solution formatting
# -----------------------------
def simple_format_general_solution(solset, var_name="x") -> str:
    if isinstance(solset, FiniteSet):
        return ", ".join(f"{var_name} = {s}" for s in solset)
    return str(solset)


# -----------------------------
# MAIN VIEW
# -----------------------------
def maths(request):
    context = {}
    func_store = request.session.get("functions", {})

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
            context["functions"] = func_store
            return render(request, "maths/maths.html", context)

        try:
            # -------------------------------
            # DEF command
            # -------------------------------
            def_cmd = extract_def_command(latex_expr)
            if def_cmd:
                fname, arg, rhs_latex = def_cmd

                body = parse_latex(rhs_latex)
                body = body.subs(Symbol("e"), exp(1)).subs(Symbol("pi"), pi)

                func_store[fname] = (arg, str(body))
                request.session["functions"] = func_store

                context["direct_result"] = f"Defined: {fname}({arg}) = {body}"
                context["functions"] = func_store
                return render(request, "maths/maths.html", context)

            # -------------------------------
            # Parse expression
            # -------------------------------
            expr = parse_latex(latex_expr)
            expr = expr.subs(Symbol("e"), exp(1)).subs(Symbol("pi"), pi)

            # Substitute defined functions: f(anything) -> body(arg=anything)
            for name, (argname, body_str) in func_store.items():
                f = Function(name)
                arg_sym = Symbol(argname)
                body_sym = sympify(body_str)
                expr = expr.replace(f, lambda a: body_sym.subs(arg_sym, a))

            context["functions"] = func_store

            # -------------------------------
            # Direct numeric evaluation
            # -------------------------------
            if not expr.free_symbols and not isinstance(expr, Relational):
                val = simplify(expr) if format_mode == "standard" else expr.evalf()
                context["direct_result"] = str(val)
                return render(request, "maths/maths.html", context)

            # -------------------------------
            # diff / integrate / antidiff typed by user
            # -------------------------------
            if expr.func.__name__ in {"diff", "integrate", "antidiff"}:
                args = expr.args
                var = sorted(expr.free_symbols, key=lambda s: s.name)[0] if expr.free_symbols else Symbol("x")

                if expr.func.__name__ == "diff":
                    result = diff(args[0], var) if len(args) == 1 else diff(*args)

                elif expr.func.__name__ == "antidiff":
                    result = integrate(args[0], var) if len(args) == 1 else integrate(*args)

                else:  # integrate(...)
                    result = integrate(*args)

                result = simplify(result)
                if format_mode == "decimal":
                    result = result.evalf()

                context["direct_result"] = str(result)
                return render(request, "maths/maths.html", context)

            # -------------------------------
            # Inequality
            # -------------------------------
            if isinstance(expr, Relational) and not isinstance(expr, Eq):
                symbols = sorted(expr.free_symbols, key=lambda s: s.name)
                var = symbols[0] if symbols else Symbol("x")

                sol = solve_univariate_inequality(expr, var, domain=S.Reals, relational=False)
                context["inequality_solution"] = str(sol)
                return render(request, "maths/maths.html", context)

            # -------------------------------
            # Equation solving
            # -------------------------------
            equation = expr if isinstance(expr, Eq) else Eq(expr, 0)
            context["equation"] = str(equation)

            # variable
            symbols = sorted(equation.free_symbols, key=lambda s: s.name)
            var = symbols[0] if symbols else Symbol("x")

            f_expr = simplify(equation.lhs - equation.rhs)

            # 1) symbolic solve attempt
            solset = solveset(f_expr, var, domain=S.Reals)

            # If SymPy gives a nice infinite family, show it
            if not isinstance(solset, FiniteSet) and not isinstance(solset, ConditionSet):
                context["general_solution"] = simple_format_general_solution(solset, str(var))
                return render(request, "maths/maths.html", context)

            # 2) numeric solve (finite set OR conditionset)
            dom_min_sym = sympify(domain_min)
            dom_max_sym = sympify(domain_max)

            a, b, used_adaptive = adaptive_numeric_solve(f_expr, var, dom_min_sym, dom_max_sym)

            # If we can't get finite numeric bounds, fall back to symbolic display
            if a is None or b is None:
                context["general_solution"] = f"Solution set described by: {solset}"
                return render(request, "maths/maths.html", context)

            f_num = f_numeric_from_sympy(f_expr, var)
            brackets = bracket_roots(f_num, a, b, N=1400)
            roots = []
            for left, right in brackets:
                r = bisect_root(f_num, left, right)
                if r is not None:
                    roots.append(r)

            roots = clean_roots(roots, eps=1e-6)

            if not roots:
                # If user asked infinite and we tried adaptively, show symbolic info instead of "none"
                if used_adaptive:
                    context["general_solution"] = f"Solution set described by: {solset}"
                else:
                    context["error"] = "No numeric solutions found in the given domain."
                return render(request, "maths/maths.html", context)

            # display
            out = []
            for r in roots:
                if format_mode == "decimal":
                    out.append(f"{var} ≈ {round(r, 10)}")
                else:
                    out.append(f"{var} = {nsimplify(r)}")

            context["solutions"] = out
            return render(request, "maths/maths.html", context)

        except Exception as e:
            context["error"] = f"Could not understand the expression: {e}"
            context["functions"] = func_store
            return render(request, "maths/maths.html", context)

    context["functions"] = func_store
    return render(request, "maths/maths.html", context)
