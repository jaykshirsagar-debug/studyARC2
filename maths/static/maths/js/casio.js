// static/maths/js/casio.js
document.addEventListener("DOMContentLoaded", function () {
    if (typeof MathQuill === "undefined") {
        console.error("MathQuill is NOT loaded");
        return;
    }

    var MQ = MathQuill.getInterface(2);
    var mathFieldElement = document.getElementById("math-field");
    var hiddenInput = document.getElementById("math-latex");

    if (!mathFieldElement) {
        console.error("No #math-field element found");
        return;
    }

    var mathField = MQ.MathField(mathFieldElement, {
        spaceBehavesLikeTab: true,
        handlers: {
            edit: function () {
                hiddenInput.value = mathField.latex();
            }
        }
    });

    // Insert helper
    function insertIntoMathField(latex) {
        mathField.write(latex);
        mathField.focus();
    }

    // Button handling
    document.querySelectorAll(".mq-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            let cmd = btn.dataset.insert;
            insertIntoMathField(cmd);
        });
    });
});
