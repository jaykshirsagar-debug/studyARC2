
document.addEventListener("DOMContentLoaded", function () {
    if (typeof MathQuill === "undefined") {
        console.error("MathQuill is NOT loaded");
        return;
    }

    var MQ = MathQuill.getInterface(2);
    var mathFieldElement = document.getElementById("math-field");
    var hiddenInput = document.getElementById("math-latex");

    var mathField = MQ.MathField(mathFieldElement, {
        spaceBehavesLikeTab: true,
        handlers: {
            edit: function () {
                hiddenInput.value = mathField.latex();
                console.log("Current LaTeX:", hiddenInput.value);
            }
        }
    });

    // Optional: prefill from last input (if you want)
    var inputLatex = document.getElementById("input-latex-data");
    if (inputLatex && inputLatex.textContent) {
        mathField.latex(inputLatex.textContent);
        hiddenInput.value = inputLatex.textContent;
    }
});
