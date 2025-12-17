# maths/views.py
from django.shortcuts import render
from django.http import JsonResponse
from sympy import (
    Eq, Symbol, exp, S, solveset, sympify,
    pi, simplify, nsimplify, Function, diff, integrate
)
from sympy.parsing.latex import parse_latex
from sympy.sets import FiniteSet, ConditionSet
import re
import json
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
# GRAPH: evaluation + adaptive sampling (Phase 4)
# =====================================================
def eval_point(expr, var, x, y_clip=1e6):
    try:
        y = expr.subs(var, x).evalf()
        y = float(y)
        if not np.isfinite(y) or abs(y) > y_clip:
            return None
        return y
    except Exception:
        return None


def adaptive_sample(expr, var, xmin, xmax, base_segments=240, max_depth=10, y_clip=1e6):
    """
    Adaptive sampling: refines segments where midpoint deviates from linear interpolation.
    Returns (xs, ys) lists (ys contains None for gaps).
    """
    xmin = float(xmin)
    xmax = float(xmax)

    xs_out = []
    ys_out = []

    def refine(x0, y0, x1, y1, depth):
        # Always output left endpoint once; right endpoint later by caller chaining
        if depth <= 0:
            return [(x0, y0), (x1, y1)]

        # If either endpoint is invalid, don't refine; keep as is (gap preserved)
        if y0 is None or y1 is None:
            return [(x0, y0), (x1, y1)]

        xm = 0.5 * (x0 + x1)
        ym = eval_point(expr, var, xm, y_clip=y_clip)

        # If midpoint invalid, keep as coarse; this helps discontinuities stay broken
        if ym is None:
            return [(x0, y0), (x1, y1)]

        ylin = 0.5 * (y0 + y1)
        scale = max(1.0, abs(y0), abs(y1), abs(ym))
        tol = 0.0025 * scale  # tweakable: smaller -> more points

        if abs(ym - ylin) > tol:
            left = refine(x0, y0, xm, ym, depth - 1)
            right = refine(xm, ym, x1, y1, depth - 1)
            return left[:-1] + right  # avoid duplicate xm
        else:
            return [(x0, y0), (x1, y1)]

    # initial coarse grid
    xs0 = np.linspace(xmin, xmax, base_segments + 1)
    ys0 = [eval_point(expr, var, float(x), y_clip=y_clip) for x in xs0]

    pairs = []
    for i in range(base_segments):
        x0, x1 = float(xs0[i]), float(xs0[i+1])
        y0, y1 = ys0[i], ys0[i+1]
        seg = refine(x0, y0, x1, y1, max_depth)
        if i > 0:
            seg = seg[1:]  # avoid duplicating previous endpoint
        pairs.extend(seg)

    # Post-process: break across huge jumps (avoid connecting over asymptotes)
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    for i in range(1, len(ys)):
        if ys[i-1] is None or ys[i] is None:
            continue
        if abs(ys[i] - ys[i-1]) > 2e5:  # jump threshold
            ys[i] = None

    return xs, ys


def normalize_graph_expr(s: str, func_store: dict) -> str:
    s = s.strip()
    if re.fullmatch(r"[A-Za-z_]\w*", s) and s in func_store:
        arg, _body = func_store[s]
        return f"{s}({arg})"
    return s


def choose_adaptive_settings(xmin, xmax):
    width = abs(float(xmax) - float(xmin))
    if width < 1:
        return 320, 11
    if width < 5:
        return 260, 10
    if width < 20:
        return 220, 10
    return 200, 9


def find_intersections(exprA, exprB, var, xmin, xmax, max_roots=20):
    """
    Numeric intersections for two expressions in [xmin, xmax].
    Uses sign-change bracketing + bisection. Returns list of (x, y).
    """
    xmin = float(xmin)
    xmax = float(xmax)

    def h(x):
        ya = eval_point(exprA, var, x)
        yb = eval_point(exprB, var, x)
        if ya is None or yb is None:
            return None
        return ya - yb

    # sample grid for bracketing
    N = 2400
    xs = np.linspace(xmin, xmax, N)
    hs = []
    for x in xs:
        hs.append(h(float(x)))

    brackets = []
    for i in range(N - 1):
        h0, h1 = hs[i], hs[i+1]
        if h0 is None or h1 is None:
            continue
        if abs(h0) < 1e-8:
            brackets.append((float(xs[i]) - (xmax - xmin) / N, float(xs[i]) + (xmax - xmin) / N))
            continue
        if h0 * h1 < 0:
            brackets.append((float(xs[i]), float(xs[i+1])))

    def bisect_h(a, b):
        fa = h(a)
        fb = h(b)
        if fa is None or fb is None:
            return None
        if fa * fb > 0:
            return None
        for _ in range(80):
            m = 0.5 * (a + b)
            fm = h(m)
            if fm is None:
                return None
            if abs(fm) < 1e-10 or abs(b - a) < 1e-10:
                return m
            if fa * fm < 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return 0.5 * (a + b)

    roots = []
    for (a, b) in brackets:
        r = bisect_h(a, b)
        if r is None:
            continue
        roots.append(r)

    # dedupe (close roots)
    roots = sorted(roots)
    uniq = []
    eps = (xmax - xmin) * 1e-4 + 1e-6
    for r in roots:
        if not uniq or abs(r - uniq[-1]) > eps:
            uniq.append(r)

    points = []
    for r in uniq[:max_roots]:
        y = eval_point(exprA, var, r)
        if y is None:
            continue
        points.append((float(r), float(y)))

    return points


def build_plot_payload(exprs, func_store, xmin, xmax):
    labels = exprs[:]

    parsed = []
    for e in exprs:
        e0 = parse_latex(e)
        e0 = e0.subs(Symbol("e"), exp(1)).subs(Symbol("pi"), pi)
        e0 = substitute_defined_functions(e0, func_store)
        parsed.append(simplify(e0))

    var = Symbol("x")
    for e in parsed:
        if e.free_symbols:
            var = sorted(list(e.free_symbols), key=lambda s: s.name)[0]
            break

    base_segments, max_depth = choose_adaptive_settings(xmin, xmax)

    series = []
    for i, e in enumerate(parsed):
        xs, ys = adaptive_sample(e, var, xmin, xmax, base_segments=base_segments, max_depth=max_depth)
        series.append({
            "name": labels[i] if i < len(labels) else f"y{i+1}",
            "x": xs,
            "y": ys,
        })

    payload = {
        "xmin": float(xmin),
        "xmax": float(xmax),
        "var": str(var),
        "series": series,
        "intersections": []
    }

    # Phase 4: intersections for exactly 2 series
    if len(parsed) == 2:
        pts = find_intersections(parsed[0], parsed[1], var, xmin, xmax, max_roots=20)
        payload["intersections"] = [{"x": x, "y": y} for (x, y) in pts]

    return payload


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
            # DEF
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

            # GRAPH
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
                        normalize_graph_expr(parts[1], func_store),
                    ]
                    xmin = float(sympify(parts[2]))
                    xmax = float(sympify(parts[3]))
                else:
                    raise ValueError("Invalid graph syntax")

                payload = build_plot_payload(exprs, func_store, xmin, xmax)

                context["plot_data_json"] = json.dumps(payload)
                context["direct_result"] = f"Graph window: [{xmin}, {xmax}]"
                return render(request, "maths/maths.html", context)

            # DIFF
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

            # INTEGRATE
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

            # NORMAL PARSE
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


# =====================================================
# PHASE 3/4: graph-data endpoint (now includes intersections + adaptive sampling)
# =====================================================
def graph_data(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        body = json.loads(request.body.decode("utf-8"))
        exprs = body.get("exprs", [])
        xmin = float(body.get("xmin"))
        xmax = float(body.get("xmax"))

        func_store = request.session.get("functions", {})

        payload = build_plot_payload(exprs, func_store, xmin, xmax)
        return JsonResponse(payload)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)
