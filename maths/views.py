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


# =====================================================
# NORMALISE MathQuill letter-by-letter CAS commands
# =====================================================
def normalize_cas(latex: str) -> str:
    latex = latex.replace(r"\left", "").replace(r"\right", "")
    latex = latex.replace(r"\ ", "")
    latex = re.sub(r"i\s*n\s*t\s*e\s*g\s*r\s*a\s*t\s*e", "integrate", latex, flags=re.I)
    latex = re.sub(r"d\s*i\s*f\s*f", "diff", latex, flags=re.I)
    latex = re.sub(r"a\s*n\s*t\s*i\s*d\s*i\s*f\s*f", "antidiff", latex, flags=re.I)
    return latex


# =====================================================
# DEF f(x)=... parser
# =====================================================
def extract_def_command(raw):
    s = raw.strip()
    if not s.lower().startswith("def"):
        return None

    if "=" not in s:
        raise ValueError("Use: def f(x)=...")

    left, rhs = s.split("=", 1)
    left = left[3:].replace("\\", "")
    left = re.sub(r"\s+", "", left)

    if "(" not in left or ")" not in left:
        raise ValueError("Use: def f(x)=...")

    fname = left.split("(")[0]
    arg = left[left.index("(") + 1:left.index(")")]

    if not fname.isidentifier():
        raise ValueError("Function name must be a valid identifier.")
    if not arg.isidentifier():
        raise ValueError("Function argument must be a variable.")

    return fname, arg, rhs


# =====================================================
# Numeric helpers (IMPROVED root capture)
# =====================================================
def f_numeric(expr, var):
    return lambda x: float(expr.subs(var, x).evalf())


def bracket_roots(f, a, b, N=6000, zero_eps=1e-6):
    """
    Better bracketing:
    - captures classic sign-changes
    - ALSO captures "near-zero" sample points (otherwise you miss roots)
    """
    xs = [a + (b - a) * i / N for i in range(N + 1)]
    out = []

    for i in range(N):
        x0, x1 = xs[i], xs[i + 1]
        try:
            f0, f1 = f(x0), f(x1)
        except Exception:
            continue

        # near-zero sample (float rarely equals exactly 0)
        if abs(f0) < zero_eps:
            h = (b - a) / N
            out.append((x0 - h, x0 + h))
            continue

        # sign change
        if f0 * f1 < 0:
            out.append((x0, x1))

    return out


def bisect(f, a, b, tol=1e-10):
    """
    More robust bisection:
    - accept near-zero endpoints
    - more iterations for stability
    """
    try:
        fa, fb = f(a), f(b)
    except Exception:
        return None

    if abs(fa) < tol:
        return a
    if abs(fb) < tol:
        return b
    if fa * fb > 0:
        return None

    for _ in range(120):
        m = (a + b) / 2.0
        try:
            fm = f(m)
        except Exception:
            return None

        if abs(fm) < tol or abs(b - a) < tol:
            return m

        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm

    return (a + b) / 2.0


def expand_domain(expr, var):
    """
    Casio-style auto-expand window until roots stabilize.
    Uses improved bracketing so you don't miss roots like x^3 = sin(x).
    """
    f = f_numeric(expr, var)
    L = 10.0
    prev = None

    for _ in range(7):
        roots = []
        for a, b in bracket_roots(f, -L, L, N=6000, zero_eps=1e-6):
            r = bisect(f, a, b, tol=1e-10)
            if r is not None:
                roots.append(r)

        # de-dup
        roots = sorted(roots)
        cleaned = []
        for r in roots:
            if not cleaned or abs(r - cleaned[-1]) > 1e-6:
                cleaned.append(r)

        key = tuple(round(r, 8) for r in cleaned)
        if key == prev:
            return cleaned

        prev = key
        L *= 2.0

    return list(prev) if prev else []


# =====================================================
# MAIN VIEW
# =====================================================
def maths(request):
    context = {}
    func_store = request.session.get("functions", {})

    if request.method == "POST":
        raw = (request.POST.get("expression") or "").strip()
        raw = normalize_cas(raw)
        format_mode = request.POST.get("format_mode", "decimal")

        context["input_latex"] = raw
        context["functions"] = func_store

        try:
            # -------------------------------
            # DEF
            # -------------------------------
            cmd = extract_def_command(raw)
            if cmd:
                name, arg, rhs = cmd
                body = parse_latex(rhs)
                body = body.subs(Symbol("e"), exp(1)).subs(Symbol("pi"), pi)
                func_store[name] = (arg, str(body))
                request.session["functions"] = func_store
                context["direct_result"] = f"Defined: {name}({arg}) = {body}"
                return render(request, "maths/maths.html", context)

            # -------------------------------
            # INTEGRATE (TEXT-FIRST, CASIO STYLE)
            # -------------------------------
            if raw.lower().startswith("integrate"):
                m = re.match(r"integrate\((.*)\)$", raw)
                if not m:
                    raise ValueError("Use integrate(f(x)) or integrate(f(x),0,1)")

                parts = [p.strip() for p in m.group(1).split(",")]
                e0 = parse_latex(parts[0])

                for name, (arg, body) in func_store.items():
                    f = Function(name)
                    e0 = e0.replace(f, lambda x: sympify(body).subs(Symbol(arg), x))

                var = sorted(e0.free_symbols, key=lambda s: s.name)[0] if e0.free_symbols else Symbol("x")

                if len(parts) == 1:
                    res = integrate(e0, var)
                elif len(parts) == 3:
                    res = integrate(e0, (var, sympify(parts[1]), sympify(parts[2])))
                else:
                    raise ValueError("Use integrate(f(x)) or integrate(f(x),0,1)")

                res = simplify(res)
                context["direct_result"] = str(res.evalf() if format_mode == "decimal" else res)
                return render(request, "maths/maths.html", context)

            # -------------------------------
            # DIFF
            # -------------------------------
            if raw.lower().startswith("diff"):
                m = re.match(r"diff\((.*)\)\s*$", raw, flags=re.I)
                if not m:
                    raise ValueError("Use diff(f(x))")

                inner = m.group(1)
                e0 = parse_latex(inner)
                e0 = e0.subs(Symbol("e"), exp(1)).subs(Symbol("pi"), pi)

                for name, (arg, body) in func_store.items():
                    f = Function(name)
                    e0 = e0.replace(f, lambda x: sympify(body).subs(Symbol(arg), x))

                var = sorted(e0.free_symbols, key=lambda s: s.name)[0] if e0.free_symbols else Symbol("x")
                res = simplify(diff(e0, var))

                context["direct_result"] = str(res.evalf() if format_mode == "decimal" else res)
                return render(request, "maths/maths.html", context)

            # -------------------------------
            # PARSE NORMAL EXPRESSION
            # -------------------------------
            expr = parse_latex(raw)
            expr = expr.subs(Symbol("e"), exp(1)).subs(Symbol("pi"), pi)

            for name, (arg, body) in func_store.items():
                f = Function(name)
                expr = expr.replace(f, lambda x: sympify(body).subs(Symbol(arg), x))

            # Inequality
            if isinstance(expr, Relational) and not isinstance(expr, Eq):
                var = sorted(expr.free_symbols, key=lambda s: s.name)[0]
                sol = solve_univariate_inequality(expr, var, domain=S.Reals)
                context["inequality_solution"] = str(sol)
                return render(request, "maths/maths.html", context)

            # Expression only
            if not isinstance(expr, Eq):
                shown = simplify(expr)
                context["direct_result"] = str(shown.evalf() if format_mode == "decimal" else shown)
                return render(request, "maths/maths.html", context)

            # Equation
            equation = expr
            var = sorted(equation.free_symbols, key=lambda s: s.name)[0] if equation.free_symbols else Symbol("x")
            f_expr = simplify(equation.lhs - equation.rhs)

            solset = solveset(f_expr, var, domain=S.Reals)
            if not isinstance(solset, (FiniteSet, ConditionSet)):
                context["general_solution"] = str(solset)
                return render(request, "maths/maths.html", context)

            # Numeric roots (auto-expand)
            roots = expand_domain(f_expr, var)
            if not roots:
                context["general_solution"] = str(solset)
                return render(request, "maths/maths.html", context)

            context["solutions"] = [
                f"{var} = {nsimplify(r)}" if format_mode == "standard" else f"{var} ≈ {round(float(r), 10)}"
                for r in roots
            ]
            return render(request, "maths/maths.html", context)

        except Exception as e:
            context["error"] = f"Could not understand the expression: {e}"

    context["functions"] = func_store
    return render(request, "maths/maths.html", context)
