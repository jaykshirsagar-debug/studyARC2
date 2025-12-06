# maths/views.py
from django.shortcuts import render
from sympy import Eq
from sympy.parsing.latex import parse_latex
from sympy import solve  # classic solver


def maths(request):
    context = {}

    if request.method == "POST":
        latex_expr = request.POST.get("expression", "").strip()
        context["input_latex"] = latex_expr  # so we can show/debug it

        if latex_expr:
            try:
                # Parse LaTeX to SymPy expression
                expr = parse_latex(latex_expr)

                # If it's not already an equation, treat it as expr = 0
                if isinstance(expr, Eq):
                    equation = expr
                else:
                    equation = Eq(expr, 0)

                # Pick a variable to solve for (first free symbol)
                symbols = sorted(equation.free_symbols, key=lambda s: s.name)
                if not symbols:
                    context["error"] = "No variable found to solve for."
                else:
                    var = symbols[0]
                    sols = solve(equation, var)  # returns a list

                    context["equation"] = str(equation)
                    context["variable"] = str(var)
                    context["solutions"] = [str(s) for s in sols]

            except Exception as e:
                context["error"] = f"Could not understand the expression: {e}"

        else:
            context["error"] = "Please enter an equation."

    return render(request, "maths/maths.html", context)