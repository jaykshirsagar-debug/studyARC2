# maths/views.py
from django.shortcuts import render
from sympy import (
    Eq, Symbol, exp, S, solveset, sympify,
    pi, simplify, nsimplify, Function, diff, integrate
)
from sympy.parsing.latex import parse_latex
from sympy.sets import FiniteSet, ConditionSet
import re
import json

# ===== GRAPHING IMPORTS (PHASE 1/2) =====
import numpy as np


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
# Replace stored functions into any SymPy expression
# =====================================================
def substitute_defined_functions(expr, func_store):
    """
    Replace f(anything) with the stored body, substituting the argument.
    """
    for name, (arg, body) in func_store.items():
        f = Function(name)
        arg_sym = Symbol(arg)
        body_sym = sympify(body)  # stored as string
        expr = expr.replace(f, lambda x: body_sym.subs(arg_sym, x))
    return expr


# =====================================================
# Numeric helpers (roots)
# =====================================================
def f_numeric(expr, var):
    return lambda x: float(expr.subs(var, x).evalf())


def bracket_roots(f, a, b, N=6000, zero_eps=1e-6):
    xs = [a + (b - a) * i / N for i in range(N + 1)]
    out = []

    for i in range(N):
        x0, x1 = xs[i], xs[i + 1]
        try:
            f0, f1 = f(x0), f(x1)
        except Exception:
            continue

        if abs(f0) < zero_eps:
            h = (b - a) / N
            out.append((x0 - h, x0 + h))
        elif f0 * f1 < 0:
            out.append((x0, x1))

    return out


def bisect(f, a, b, tol=1e-10):
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
    f = f_numeric(expr, var)
    L = 10.0
    prev = None

    for _ in range(7):
        roots = []
        for a, b in bracket_roots(f, -L, L):
            r = bisect(f, a, b)
            if r is not None:
                roots.append(r)

        roots = sorted(set(round(r, 6) for r in roots))
        if roots == prev:
            return roots

        prev = roots
        L *= 2.0

    return prev or []


# =====================================================
# GRAPH HELPERS (compute-only)
# =====================================================
def eval_series(expr, var, xs, y_clip=1e6):
    ys = []
    for x in xs:
        try:
            y = expr.subs(var, x).evalf()
            y = float(y)
            if not np.isfinite(y) or abs(y) > y_clip:
                ys.append(np.nan)
            else:
                ys.append(y)
        except Exception:
            ys.append(np.nan)
    return np.array(ys, dtype=float)


def compute_series(expr, var, xmin, xmax, N=1200, y_clip=1e6):
    xs = np.linspace(float(xmin), float(xmax), int(N))
    ys = eval_series(expr, var, xs, y_clip=y_clip)

    # Break lines across large jumps (asymptote-ish)
    if len(ys) > 2:
        dy = np.abs(np.diff(ys))
        jump = dy > (0.2 * y_clip)
        ys2 = ys.copy()
        ys2[1:][jump] = np.nan
    else:
        ys2 = ys

    return xs, ys2


def compute_plot_data(parsed_exprs, labels, var, xmin, xmax, N=1200):
    series = []
    for i, e in enumerate(parsed_exprs):
        xs, ys = compute_series(e, var, xmin, xmax, N=N)
        label = labels[i] if i < len(labels) else f"y{i+1}"
        series.append({
            "name": label,
            "x": xs,
            "y": ys,
        })
    return {
        "xmin": float(xmin),
        "xmax": float(xmax),
        "var": str(var),
        "series": series,
    }


def normalize_graph_expr(s: str, func_store: dict) -> str:
    s = s.strip()
    # If user typed just "g" and g is a defined function, convert to "g(x)"
    if re.fullmatch(r"[A-Za-z_]\w*", s) and s in func_store:
        arg, _body = func_store[s]
        return f"{s}({arg})"
    return s


def np_to_jsonable(arr):
    # Plotly is happy with nulls for gaps; convert NaN -> None
    out = []
    for v in arr.tolist():
        if v is None:
            out.append(None)
        else:
            try:
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    out.append(None)
                else:
                    out.append(v)
            except Exception:
                out.append(None)
    return out


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
        context["format_mode"] = format_mode

        try:
            # -------------------------------
            # DEF
            # -------------------------------
            cmd = extract_def_command(raw)
            if cmd:
                name, arg, rhs = cmd

                body = parse_latex(rhs)
                body = body.subs(Symbol("e"), exp(1)).subs(Symbol("pi"), pi)
                body = substitute_defined_functions(body, func_store)

                func_store[name] = (arg, str(body))
                request.session["functions"] = func_store

                context["functions"] = func_store
                context["direct_result"] = f"Defined: {name}({arg}) = {body}"
                return render(request, "maths/maths.html", context)

            # -------------------------------
            # GRAPH (PHASE 2: Plotly)
            # -------------------------------
            if raw.lower().startswith("graph"):
                m = re.match(r"graph\((.*)\)\s*$", raw, flags=re.I)
                if not m:
                    raise ValueError("Use graph(f(x)) or graph(f(x),-10,10)")

                parts = [p.strip() for p in m.group(1).split(",")]
                xmin, xmax = -10, 10

                if len(parts) == 1:
                    exprs = [normalize_graph_expr(parts[0], func_store)]
                elif len(parts) == 3:
                    exprs = [normalize_graph_expr(parts[0], func_store)]
                    xmin = float(sympify(parts[1]))
                    xmax = float(sympify(parts[2]))
                elif len(parts) == 4:
                    exprs = [
                        normalize_graph_expr(parts[0], func_store),
                        normalize_graph_expr(parts[1], func_store)
                    ]
                    xmin = float(sympify(parts[2]))
                    xmax = float(sympify(parts[3]))
                else:
                    raise ValueError("Invalid graph syntax")

                # Labels shown in legend (use the expression string)
                labels = exprs[:]  # already normalized (e.g., g -> g(x))

                parsed = []
                for e in exprs:
                    e0 = parse_latex(e)
                    e0 = e0.subs(Symbol("e"), exp(1)).subs(Symbol("pi"), pi)
                    e0 = substitute_defined_functions(e0, func_store)
                    parsed.append(simplify(e0))

                # pick plotting variable
                var = Symbol("x")
                for e in parsed:
                    if e.free_symbols:
                        var = sorted(list(e.free_symbols), key=lambda s: s.name)[0]
                        break

                plot_data = compute_plot_data(parsed, labels, var, xmin, xmax, N=1600)

                # Convert numpy arrays to JSON-safe lists (NaN -> null)
                payload = {
                    "xmin": plot_data["xmin"],
                    "xmax": plot_data["xmax"],
                    "var": plot_data["var"],
                    "series": [
                        {
                            "name": s["name"],
                            "x": np_to_jsonable(s["x"]),
                            "y": np_to_jsonable(s["y"]),
                        }
                        for s in plot_data["series"]
                    ]
                }

                context["plot_data_json"] = json.dumps(payload)
                context["direct_result"] = f"Graph window: [{xmin}, {xmax}]"
                return render(request, "maths/maths.html", context)

            # -------------------------------
            # DIFF
            # -------------------------------
            if raw.lower().startswith("diff"):
                m = re.match(r"diff\((.*)\)\s*$", raw, flags=re.I)
                if not m:
                    raise ValueError("Use diff(f(x))")

                e0 = parse_latex(m.group(1))
                e0 = e0.subs(Symbol("e"), exp(1)).subs(Symbol("pi"), pi)
                e0 = substitute_defined_functions(e0, func_store)

                var = sorted(e0.free_symbols, key=lambda s: s.name)[0] if e0.free_symbols else Symbol("x")
                res = simplify(diff(e0, var))

                context["direct_result"] = str(res.evalf() if format_mode == "decimal" else res)
                return render(request, "maths/maths.html", context)

            # -------------------------------
            # INTEGRATE
            # -------------------------------
            if raw.lower().startswith("integrate"):
                m = re.match(r"integrate\((.*)\)\s*$", raw, flags=re.I)
                if not m:
                    raise ValueError("Use integrate(f(x)) or integrate(f(x),0,1)")

                parts = [p.strip() for p in m.group(1).split(",")]

                e0 = parse_latex(parts[0])
                e0 = e0.subs(Symbol("e"), exp(1)).subs(Symbol("pi"), pi)
                e0 = substitute_defined_functions(e0, func_store)

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
            # NORMAL PARSE
            # -------------------------------
            expr = parse_latex(raw)
            expr = expr.subs(Symbol("e"), exp(1)).subs(Symbol("pi"), pi)
            expr = substitute_defined_functions(expr, func_store)

            if not isinstance(expr, Eq):
                shown = simplify(expr)
                context["direct_result"] = str(shown.evalf() if format_mode == "decimal" else shown)
                return render(request, "maths/maths.html", context)

            var = sorted(expr.free_symbols, key=lambda s: s.name)[0] if expr.free_symbols else Symbol("x")
            f_expr = simplify(expr.lhs - expr.rhs)

            solset = solveset(f_expr, var, domain=S.Reals)
            if not isinstance(solset, (FiniteSet, ConditionSet)):
                context["general_solution"] = str(solset)
                return render(request, "maths/maths.html", context)

            roots = expand_domain(f_expr, var)
            context["solutions"] = [
                f"{var} = {nsimplify(r)}" if format_mode == "standard" else f"{var} ≈ {r}"
                for r in roots
            ]
            return render(request, "maths/maths.html", context)

        except Exception as e:
            context["error"] = f"Could not understand the expression: {e}"

    context["functions"] = func_store
    context["format_mode"] = context.get("format_mode", "decimal")
    return render(request, "maths/maths.html", context)
