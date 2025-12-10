// static/maths/js/casio.js
document.addEventListener("DOMContentLoaded", function () {
    console.log("casio.js DOMContentLoaded");

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
                console.log("Current LaTeX:", hiddenInput.value);
            }
        }
    });

    // ---- INSERT COMMAND INTO MATH FIELD ----
    function insertIntoMathField(latex) {
        console.log("Inserting:", latex);
        mathField.write(latex);
        mathField.focus();
    }

    // ---- BUTTON HANDLING ----
    const buttons = document.querySelectorAll(".mq-btn");
    console.log("Found mq-btn buttons:", buttons.length);

    buttons.forEach(btn => {
        btn.addEventListener("click", () => {
            let cmd = btn.dataset.insert;
            insertIntoMathField(cmd);
        });
    });
});
