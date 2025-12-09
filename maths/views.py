# maths/views.py
from django.shortcuts import render
from sympy import (
    Eq, Symbol, exp, Interval, S, solveset, sympify,
    FiniteSet, ConditionSet, Union, nsolve, latex, I, pi, simplify
)
from sympy.parsing.latex import parse_latex
from sympy.sets.fancysets import ImageSet
import numpy as np


def format_general_solution(solset, var_name):
    from sympy import pi, Symbol, simplify
    from sympy.sets import FiniteSet, ConditionSet, Union, ImageSet

    # 1. Finite set → just list the values
    if isinstance(solset, FiniteSet):
        return ", ".join(f"{var_name} = {s}" for s in solset)

    # 2. No closed form
    if isinstance(solset, ConditionSet):
        return f"Solution set described by: {solset}"

    # Helper: extract lambda expressions from ImageSet / Union(ImageSet, ...)
    def extract_lambdas(s):
        if isinstance(s, ImageSet):
            return [s.lamda.expr]
        if isinstance(s, Union):
            exprs = []
            for subset in s.args:
                if isinstance(subset, ImageSet):
                    exprs.append(subset.lamda.expr)
            return exprs
        return []

    lam_exprs = extract_lambdas(solset)
    lam_exprs = [simplify(e) for e in lam_exprs]

    if not lam_exprs:
        return str(solset)

    # Helper: get the constant offset (mod 2π) from something like 2*pi*n + offset
    def offset_mod_2pi(expr):
        expr = simplify(expr)
        if expr.is_Add:
            for term in expr.args:
                if term.free_symbols == set():  # constant term
                    return simplify(term % (2*pi))
            return 0
        return simplify(expr % (2*pi))

    # ---------------------------
    # Case: two ImageSets (typical trig)
    # ---------------------------
    if len(lam_exprs) == 2:
        e1, e2 = lam_exprs
        off1 = offset_mod_2pi(e1)
        off2 = offset_mod_2pi(e2)

        offs = {off1, off2}

        # sin(x) = 0 → offsets 0 and π
        if offs == {0, pi}:
            return f"{var_name} = n*pi,  n ∈ ℤ"

        # cos(x) = 0 → offsets π/2 and 3π/2
        if offs == {pi/2, 3*pi/2}:
            return f"{var_name} = pi/2 + n*pi,  n ∈ ℤ"

        # General sin(x) = c or cos(x) = c:
        # e.g. {pi/6, 5*pi/6}, {α, π-α}, etc.
        # Sort for nicer output
        ordered_offs = sorted(list(offs), key=lambda o: float(o.evalf()))
        parts = [f"{var_name} = {o} + 2*pi*n" for o in ordered_offs]
        return " or ".join(parts) + ",  n ∈ ℤ"

    # ---------------------------
    # Case: single ImageSet (e.g. tan(x) = 0)
    # ---------------------------
    if len(lam_exprs) == 1:
        expr = lam_exprs[0]

        # tan(x) = 0 → n*pi (period π)
        # (and sometimes sin(x)=0 simplifies to this too)
        if simplify(expr % pi) == 0:
            return f"{var_name} = n*pi,  n ∈ ℤ"

        # Generic periodic form
        return f"{var_name} = {expr},  n ∈ ℤ"

    # Fallback
    return str(solset)



def numeric_scan_nsolve(f, var, domain_a, domain_b, solution_type="real", var_name="x"):
    """
    Try to find numeric roots of f(var) = 0 in [domain_a, domain_b] using nsolve
    from multiple initial guesses. Returns a list of LaTeX strings like 'x = ...'.
    """
    roots = set()

    # Convert domain bounds to floats (fallback to [-10,10] if something goes wrong)
    try:
        a = float(domain_a.evalf()) if hasattr(domain_a, "evalf") else float(domain_a)
        b = float(domain_b.evalf()) if hasattr(domain_b, "evalf") else float(domain_b)
    except Exception:
        a, b = -10.0, 10.0

    guess_values = np.linspace(a, b, 40)

    for g in guess_values:
        try:
            sol = nsolve(f, var, g)
            sol_c = complex(sol.evalf())
        except Exception:
            continue

        # If real-only mode, skip complex roots
        if solution_type == "real" and abs(sol_c.imag) > 1e-6:
            continue

        if abs(sol_c.imag) < 1e-6:
            real_part = round(sol_c.real, 6)
            val = sympify(real_part)
        else:
            real_part = round(sol_c.real, 6)
            imag_part = round(sol_c.imag, 6)
            val = sympify(real_part) + sympify(imag_part) * I

        eq_latex = rf"{var_name} = {latex(val)}"
        roots.add(eq_latex)

    # Sort just for stable display; real before complex roughly
    return sorted(list(roots))


def maths(request):
    context = {}

    if request.method == "POST":
        latex_expr = request.POST.get("expression", "").strip()
        solution_type = request.POST.get("solution_type", "real")    # "real" or "complex"
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
                # Parse LaTeX to SymPy expression
                expr = parse_latex(latex_expr)

                # Treat 'e' as Euler's constant
                expr = expr.subs(Symbol("e"), exp(1))

                # If it's not already an equation, treat it as expr = 0
                if isinstance(expr, Eq):
                    equation = expr
                else:
                    equation = Eq(expr, 0)

                # Pick a variable to solve for (first free symbol)
                symbols = sorted(equation.free_symbols, key=lambda s: s.name)
                if not symbols:
                    context["error"] = "No variable found to solve for."
                    return render(request, "maths/maths.html", context)

                var = symbols[0]
                var_name = str(var)
                context["variable"] = var_name

                # Store equation as LaTeX for output
                context["equation"] = latex(equation)

                # Work with f(x) = 0
                f = equation.lhs - equation.rhs

                # ===== MODE 1: GENERAL SOLUTION =====
                if solve_mode == "general":
                    solset = solveset(f, var, domain=S.Reals)
                    formatted = format_general_solution(solset, var_name)
                    context["general_solution"] = formatted
                    return render(request, "maths/maths.html", context)

                # ===== MODE 2: SOLUTIONS IN AN INTERVAL =====
                # Interpret domain bounds (allow things like '2*pi')
                a = sympify(domain_min)
                b = sympify(domain_max)
                domain = Interval(a, b)

                solset_interval = solveset(f, var, domain=domain)

                solutions = []

                # If FiniteSet: enumerate (exact or symbolic)
                if isinstance(solset_interval, FiniteSet):
                    for s in solset_interval:
                        val = s.evalf()
                        if solution_type == "real":
                            if val.is_real:
                                solutions.append(rf"{var_name} = {latex(val)}")
                        else:
                            solutions.append(rf"{var_name} = {latex(val)}")

                # If no finite solutions or non-finite set, try numeric scan
                if not solutions and not isinstance(solset_interval, FiniteSet):
                    solutions = numeric_scan_nsolve(f, var, a, b, solution_type=solution_type, var_name=var_name)

                if solutions:
                    context["solutions"] = solutions
                else:
                    context["error"] = "No solutions found in the given domain."

            except Exception as e:
                context["error"] = f"Could not understand the expression: {e}"

        else:
            context["error"] = "Please enter an equation."

    return render(request, "maths/maths.html", context)
