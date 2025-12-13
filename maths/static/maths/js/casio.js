// static/maths/js/casio.js
document.addEventListener("DOMContentLoaded", function () {
    if (typeof MathQuill === "undefined") {
        console.error("MathQuill is NOT loaded");
        return;
    }

    var MQ = MathQuill.getInterface(2);
    var mathFieldElement = document.getElementById("math-field");
    var hiddenInput = document.getElementById("math-latex");

    if (!mathFieldElement || !hiddenInput) {
        console.error("Missing #math-field or #math-latex");
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

    // Restore previous input (server-rendered hidden input value)
    if (hiddenInput.value) {
        mathField.latex(hiddenInput.value);
    }

    function insertIntoMathField(latex) {
        mathField.write(latex);
        mathField.focus();
        hiddenInput.value = mathField.latex();
    }

    // Button handling
    document.querySelectorAll(".mq-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            let cmd = btn.dataset.insert;

            // --- Usability + SymPy-friendly templates ---
            // Natural log: insert ln( )
            if (cmd === "\\log") cmd = "\\ln\\left(\\right)";

            // sqrt and nth-root
            if (cmd === "\\sqrt") cmd = "\\sqrt{ }";
            if (cmd === "\\sqrt[]") cmd = "\\sqrt[ ]{ }";

            // fraction template
            if (cmd === "\\frac{}{}") cmd = "\\frac{ }{ }";

            // exponent template
            if (cmd === "^") cmd = "^{ }";

            // typed constants
            if (cmd === "e") cmd = "e";
            if (cmd === "\\pi") cmd = "\\pi";

            // optional: a real "space" in MathQuill (for typing "def f(x)=...")
            // Add a button with data-insert="\\space" to use this.
            if (cmd === "\\space") cmd = "\\ ";

            insertIntoMathField(cmd);
        });
    });

    // Optional: Enter key submits the form like EXE
    // (Nice Casio-feel)
    mathFieldElement.addEventListener("keydown", function (e) {
        if (e.key === "Enter") {
            e.preventDefault();
            const form = mathFieldElement.closest("form");
            if (form) form.submit();
        }
    });
});
